#include "comm_backend.h"
#include <thread>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cctype>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <cerrno>

__global__ void CheckKernel (bool* flag, float i, float* remote_buffer) {
    if (remote_buffer[0] != i + 1)
    {
        *flag = false;
    }
    else
    {
        *flag = true;
    }
}

__device__ __forceinline__ uint32_t load_sys(const volatile uint32_t* addr) {
    uint32_t val;
    // 使用 acquire + sys，保证从全局内存读取最新值
    asm volatile("ld.global.acquire.sys.u32 %0, [%1];" 
                 : "=r"(val) : "l"(addr));
    return val;
}

__global__ void RDMA_PollingKernel(uint32_t* flag0, uint32_t* flag1)
{
    if (blockIdx.x == 0)
    {
        while (load_sys(flag0) == 0);
        *flag0 = 0;
    }
    else if (blockIdx.x == 1)
    {
        while (load_sys(flag1) == 0);
        *flag1 = 0;
    }
}

__global__ void RDMA_PollingKernelSeq1(const uint32_t* flag0, uint32_t expected0)
{
    while (load_sys(flag0) < expected0);
}

__global__ void RDMA_PollingKernelSeq2(const uint32_t* flag0, const uint32_t* flag1, uint32_t expected0, uint32_t expected1)
{
    if (blockIdx.x == 0) {
        while (load_sys(flag0) < expected0);
    } else if (blockIdx.x == 1) {
        while (load_sys(flag1) < expected1);
    }
}

namespace FluidGPU {

    void pcie_handler::debug_print_buffer_addresses(int device_id, int rank) {
        cudaSetDevice(device_id);
        
        Logger::get_instance().log(LOG_INFO, "=== Buffer Address Debug Info ===");
        Logger::get_instance().log(LOG_INFO, "Device: %d, Rank: %d", device_id, rank);
        Logger::get_instance().log(LOG_INFO, "BufferCount: %d, BufferSize: %ld", BufferCount, BufferSize);
        
        // 输出本地缓冲区信息
        Logger::get_instance().log(LOG_INFO, "--- Local Buffers (buffers) ---");
        for (size_t i = 0; i < buffers.size(); ++i) {
            Logger::get_instance().log(LOG_INFO, "Buffer[%zu]: addr=%p, size=%ld, shm_id=%d, event=%p", 
                                    i, buffers[i]->buffer, buffers[i]->size, 
                                    buffers[i]->shm_id, buffers[i]->event);
        }
        
        // 输出对端接收缓冲区信息
        Logger::get_instance().log(LOG_INFO, "--- Peer Receive Buffers (peer_recv_buffers) ---");
        for (size_t i = 0; i < peer_recv_buffers.size(); ++i) {
            Logger::get_instance().log(LOG_INFO, "PeerRecvBuffer[%zu]: addr=%p, size=%ld, shm_id=%d, event=%p", 
                                    i, peer_recv_buffers[i]->buffer, peer_recv_buffers[i]->size, 
                                    peer_recv_buffers[i]->shm_id, peer_recv_buffers[i]->event);
        }
        
        Logger::get_instance().log(LOG_INFO, "=== End Buffer Debug Info ===");
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::get_instance().log(LOG_ERROR, "CUDA error during debug print: %s", cudaGetErrorString(err));
        }
    }

    void pcie_handler::setup_connection(std::string ip_port, int device_id, int rank, int worker_num)
    {
        cudaSetDevice(device_id);
        sockfd = socket_connect(ip_port, rank);
        Logger::get_instance().log(LOG_INFO, "Socket connection established to %s", ip_port.c_str());
        std::vector<std::string> SHM_NAMEs;
        for (int i = 0; i < BufferCount; ++i) {
            SHM_NAMEs.push_back("/cuda_ipc_shm_11" + std::to_string(i * 2));
            SHM_NAMEs.push_back("/cuda_ipc_shm_11" + std::to_string(i * 2 + 1));
        }
        if (rank == 0)
        {
            for (const auto& shm_name : SHM_NAMEs) {
                shm_unlink(shm_name.c_str()); // Clean up previous shm
                int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
                if (fd < 0) {
                    throw std::runtime_error("Failed to create shared memory object");
                }
                ftruncate(fd, sizeof(ShmBlock));
                ShmBlock* ptr = static_cast<ShmBlock*>(mmap(nullptr, sizeof(ShmBlock), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
                if (ptr == MAP_FAILED) {
                    throw std::runtime_error("Failed to map shared memory object");
                }
                shm_blocks.push_back(ptr);
                shm_blocks.back()->ready.store(false, std::memory_order_release);
            }
            int device_id_ = device_id;
            write(sockfd, &device_id_, sizeof(int));
            void* temp_send_buffer[BufferCount];
            for (int i = 0; i < BufferCount; ++i) {
                cudaMalloc(&temp_send_buffer[i], BufferSize);
                cudaIpcGetMemHandle(&shm_blocks[i]->memHandle, temp_send_buffer[i]); 
                cudaEvent_t temp_event;
                cudaEventCreate(&temp_event, cudaEventDisableTiming | cudaEventInterprocess);
                cudaIpcGetEventHandle(&shm_blocks[i]->eventHandle, temp_event);
                BufferBlock* buffer_block = new BufferBlock();
                buffer_block->buffer = temp_send_buffer[i];
                buffer_block->size = BufferSize;
                buffer_block->event = temp_event;
                buffer_block->shm_id = i;
                buffers.push_back(buffer_block);
            }
            bool send_buffers_ready = true;
            bool peer_send_buffers_ready;
            write(sockfd, &send_buffers_ready, sizeof(bool));
            read(sockfd, &peer_send_buffers_ready, sizeof(bool));
            if (!peer_send_buffers_ready) {
                throw std::runtime_error("Peer did not acknowledge send buffers");
            }
            for (int i = 0; i < BufferCount; ++i) {
                void* temp_recv_buffer;
                cudaIpcOpenMemHandle(&temp_recv_buffer, shm_blocks[i + BufferCount]->memHandle, cudaIpcMemLazyEnablePeerAccess);
                cudaEvent_t temp_event;
                cudaIpcOpenEventHandle(&temp_event, shm_blocks[i + BufferCount]->eventHandle);
                BufferBlock* buffer_block = new BufferBlock();
                buffer_block->buffer = temp_recv_buffer;
                buffer_block->size = BufferSize;
                buffer_block->event = temp_event;
                buffer_block->shm_id = i + BufferCount;
                peer_recv_buffers.push_back(buffer_block);
            }
            bool recv_buffers_ready = true;
            write(sockfd, &recv_buffers_ready, sizeof(bool));
            bool peer_recv_buffers_ready;
            read(sockfd, &peer_recv_buffers_ready, sizeof(bool));
            if (!peer_recv_buffers_ready) {
                throw std::runtime_error("Peer did not acknowledge recv buffers");
            }
            Logger::get_instance().log(LOG_INFO, "Producer setup complete, waiting for consumer");
        } 
        else if (rank == 1)
        {
            int remote_device_id;
            read(sockfd, &remote_device_id, sizeof(remote_device_id));
            if (remote_device_id < 0) {
                throw std::runtime_error("Producer setup failed");
            }
            cudaDeviceEnablePeerAccess(remote_device_id, 0);
            cudaSetDevice(remote_device_id);
            cudaDeviceEnablePeerAccess(device_id, 0);
            cudaSetDevice(device_id);
            for (const auto& shm_name : SHM_NAMEs) {
                int fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
                if (fd < 0) {
                    throw std::runtime_error("Failed to open shared memory object");
                }
                ShmBlock* ptr = static_cast<ShmBlock*>(mmap(nullptr, sizeof(ShmBlock), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
                if (ptr == MAP_FAILED) {
                    throw std::runtime_error("Failed to map shared memory object");
                }
                shm_blocks.push_back(ptr);
            }
            void* temp_send_buffer[BufferCount];
            for (int i = 0; i < BufferCount; ++i) 
            {
                cudaMalloc(&temp_send_buffer[i], BufferSize);
                cudaIpcGetMemHandle(&shm_blocks[i + BufferCount]->memHandle, temp_send_buffer[i]);
                cudaEvent_t temp_event;
                cudaEventCreate(&temp_event, cudaEventDisableTiming | cudaEventInterprocess);
                cudaIpcGetEventHandle(&shm_blocks[i + BufferCount]->eventHandle, temp_event);
                BufferBlock* buffer_block = new BufferBlock();
                buffer_block->buffer = temp_send_buffer[i];
                buffer_block->size = BufferSize;
                buffer_block->event = temp_event;
                buffer_block->shm_id = i + BufferCount;
                buffers.push_back(buffer_block);
            }
            bool send_buffers_ready = true;
            bool peer_send_buffers_ready;
            read(sockfd, &peer_send_buffers_ready, sizeof(bool));
            if (!peer_send_buffers_ready) {
                throw std::runtime_error("Peer did not acknowledge send buffers");
            }
            write(sockfd, &send_buffers_ready, sizeof(bool));
            for (int i = 0; i < BufferCount; ++i) {
                void* recv_buffer;
                cudaIpcOpenMemHandle(&recv_buffer, shm_blocks[i]->memHandle, cudaIpcMemLazyEnablePeerAccess);
                cudaEvent_t temp_event;
                cudaIpcOpenEventHandle(&temp_event, shm_blocks[i]->eventHandle);
                BufferBlock* buffer_block = new BufferBlock();
                buffer_block->buffer = recv_buffer;
                buffer_block->size = BufferSize;
                buffer_block->event = temp_event;
                buffer_block->shm_id = i;
                peer_recv_buffers.push_back(buffer_block);
            }
            bool recv_buffers_ready = true;
            bool peer_recv_buffers_ready;
            read(sockfd, &peer_recv_buffers_ready, sizeof(bool));
            if (!peer_recv_buffers_ready) {
                throw std::runtime_error("Peer did not acknowledge recv buffers");
            }
            write(sockfd, &recv_buffers_ready, sizeof(bool));
            Logger::get_instance().log(LOG_INFO, "Consumer setup complete");
        }

        // this->debug_print_buffer_addresses(device_id, rank);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::get_instance().log(LOG_ERROR, "CUDA error during setup connection: %s", cudaGetErrorString(err));
            throw std::runtime_error("CUDA error during setup connection");
        }
    }

    void pcie_handler::test_connection(int device_id, int rank) {
        cudaSetDevice(device_id);
        cudaStream_t temp_stream[BufferCount];
        for (int i = 0; i < BufferCount; i++) {
            cudaStreamCreate(&temp_stream[i]);
        }
        bool* test_flag;
        cudaMallocHost(&test_flag, sizeof(bool));
        float test_buffers[BufferCount];
        for (int i = 0; i < BufferCount; i++) {
            test_buffers[i] = static_cast<float>(i + 1);
            cudaMemcpyAsync(peer_recv_buffers[i]->buffer, &test_buffers[i], sizeof(float), cudaMemcpyHostToDevice, temp_stream[i]);
            cudaEventRecord(peer_recv_buffers[i]->event, temp_stream[i]);
            shm_blocks[peer_recv_buffers[i]->shm_id]->ready.store(true, std::memory_order_release);
        }
        for (int i = 0; i < BufferCount; ++i) {
            while (shm_blocks[buffers[i]->shm_id]->ready.load(std::memory_order_acquire) != true) {
                __asm__ __volatile__("pause");
            }
            cudaStreamWaitEvent(temp_stream[i], buffers[i]->event, 0);
            CheckKernel<<<1, 1, 0, temp_stream[i]>>>(test_flag, i, (float*)(buffers[i]->buffer));
            cudaStreamSynchronize(temp_stream[i]);
            if (test_flag[0] == false)
            {
                Logger::get_instance().log(LOG_ERROR, "Data mismatch in block %d for device %d, rank %d", i, device_id, rank);
                throw std::runtime_error("Data mismatch during connection test");
            }

        }
        for (int i = 0; i < BufferCount; ++i) {
            shm_blocks[buffers[i]->shm_id]->ready.store(false, std::memory_order_release);
            buffers[i]->size = 0;
        }
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::get_instance().log(LOG_ERROR, "CUDA error during test connection: %s", cudaGetErrorString(err));
            throw std::runtime_error("CUDA error during test connection");
        }
        Logger::get_instance().log(LOG_INFO, "Connection test successful for device %d, rank %d", device_id, rank);
        std::cout << "Connection test successful for device " << device_id << ", rank " << rank << std::endl;
        cudaFreeHost(test_flag);
    }

    void pcie_handler::getRecvBuffers(void** temp_buffers, int buffer_num, cudaStream_t stream) {
        if (buffer_num <= 0 || buffer_num > buffers.size()) {
            throw std::out_of_range("Invalid buffer number");
        }
        for (int i = 0; i < buffer_num; ++i) {
            while (shm_blocks[buffers[i]->shm_id]->ready.load(std::memory_order_acquire) != true) {
                __asm__ __volatile__("pause");
            }
            cudaEventSynchronize(buffers[i]->event);
            shm_blocks[buffers[i]->shm_id]->ready.store(false, std::memory_order_release);
            *reinterpret_cast<void**>(temp_buffers[i]) = buffers[i]->buffer;
        }
    }

    void pcie_handler::getSendBuffers(void** buffers, int buffer_num) {
        if (buffer_num < 0 || buffer_num >= peer_recv_buffers.size()) {
            throw std::out_of_range("Invalid buffer ID");
        }
        for (int i = 0; i < buffer_num; i++)
        {
            *reinterpret_cast<void**>(buffers[i]) = peer_recv_buffers[i]->buffer;
        }
    }

    void pcie_handler::transferBufferPeer(void* pointer, int64_t size, int buffer_id, int device_id, cudaStream_t stream) 
    {
        cudaSetDevice(device_id);
        if (buffer_id < 0 || buffer_id >= BufferCount || size <= 0 || size > BufferSize) {
            throw std::invalid_argument("Invalid size for buffer transfer");
        }
        void* send_buffer = peer_recv_buffers[buffer_id]->buffer;
        cudaMemcpyAsync(send_buffer, *reinterpret_cast<void**>(pointer), size, cudaMemcpyDeviceToDevice, stream);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::get_instance().log(LOG_ERROR, "CUDA error during buffer transfer: %s", cudaGetErrorString(err));
            throw std::runtime_error("CUDA error during buffer transfer");
        }
    }

    void pcie_handler::Record(int buffer_id, int device_id, cudaStream_t stream) 
    {
        cudaSetDevice(device_id);
        if (buffer_id < 0 || buffer_id >= BufferCount) {
            throw std::invalid_argument("Invalid buffer number for recording");
        }

        cudaEventRecord(peer_recv_buffers[buffer_id]->event, stream);
        shm_blocks[peer_recv_buffers[buffer_id]->shm_id]->ready.store(true, std::memory_order_release);
    }

    double pcie_handler::communicate_test(void* pointer, int64_t size, int rank, int device_id) 
    {
        cudaSetDevice(device_id);
        (void)rank;
        if (pointer == nullptr || size <= 0 || size > BufferSize) {
            throw std::invalid_argument("Invalid size for communication test");
        }

        void* recv_buffer = peer_recv_buffers[0]->buffer;
        for (int i = 0; i < 5; i++)
        {
            cudaMemcpyPeer(recv_buffer, 1, pointer, 0, size);
            cudaDeviceSynchronize();
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; i++)
        {
            cudaMemcpyPeer(recv_buffer, 1, pointer, 0, size);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        return elapsed.count() / 100.0;
    }

    bool pcie_handler::barrier(int rank) 
    {
        if (rank == 0)
        {
            bool barrier = true;
            write(sockfd, &barrier, sizeof(bool));
            bool peer_barrier;
            read(sockfd, &peer_barrier, sizeof(bool));
            if (!peer_barrier) {
                perror("Peer did not acknowledge barrier");
                return false;
            }
        }
        else if (rank == 1)
        {
            bool peer_barrier;
            read(sockfd, &peer_barrier, sizeof(bool));
            if (!peer_barrier) {
                perror("Peer did not acknowledge barrier");
                return false;
            }
            bool barrier = true;
            write(sockfd, &barrier, sizeof(bool));
        }
        return true;
    }

    static int parse_nic_index(const char* name)
    {
        if (!name) return std::numeric_limits<int>::max();
        std::string s(name);
        const size_t pos = s.find_last_of('_');
        if (pos == std::string::npos || pos + 1 >= s.size()) return std::numeric_limits<int>::max();
        for (size_t i = pos + 1; i < s.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
                return std::numeric_limits<int>::max();
            }
        }
        return std::stoi(s.substr(pos + 1));
    }

    std::vector<ibv_device*> rdma_handler::Select_IB_Devices(int required_bandwidth_gbps, int rank) 
    {
        int num_devices = 0;
        ibv_device** dev_list = ibv_get_device_list(&num_devices);
        if (!dev_list) {
            throw std::runtime_error("Failed to get RDMA devices list");
        }

        std::vector<ibv_device*> ib_devs;

        for (int i = 0; i < num_devices; ++i) {
            ibv_device* device = dev_list[i];
            ibv_context* context = ibv_open_device(device);
            if (!context) continue;

            ibv_port_attr port_attr;
            if (ibv_query_port(context, 1, &port_attr) == 0) {
                if (port_attr.state == IBV_PORT_ACTIVE) {
                    // active_speed: 1=2.5Gbps, 2=5Gbps, 4=10Gbps, 8=20Gbps, 16=25Gbps, 32=50Gbps
                    // active_width: 1=x1, 2=x2, 4=x4, 8=x8, 16=x12
                    int speed_gbps = 0;
                    switch (port_attr.active_speed) {
                        
                        case 1: speed_gbps = 2.5; break;
                        case 2: speed_gbps = 5; break;
                        case 4: speed_gbps = 10; break;
                        case 8: speed_gbps = 20; break;
                        case 16: speed_gbps = 25; break;
                        case 32: speed_gbps = 50; break;
                        case 64: speed_gbps = 100; break;
                        default: speed_gbps = 0; break;
                    }

                    int total_bw = speed_gbps * port_attr.active_width; // Gbps
                    if (total_bw >= required_bandwidth_gbps) {
                        ib_devs.push_back(device);
                    }
                }
            }
            ibv_close_device(context);
        }

        // Optional per-rank NIC pinning via environment:
        //   FLUIDGPU_RDMA_NICS_RANK0=<hca0>,<hca1>
        //   FLUIDGPU_RDMA_NICS_RANK1=<hca0>,<hca1>
        // If unset, fall back to sorted active NICs.
        if (rank == 0 || rank == 1) {
            const char* env_value = std::getenv(rank == 0 ? "FLUIDGPU_RDMA_NICS_RANK0" : "FLUIDGPU_RDMA_NICS_RANK1");
            std::vector<std::string> preferred_names;
            if (env_value && std::strlen(env_value) > 0) {
                std::stringstream ss(env_value);
                std::string name;
                while (std::getline(ss, name, ',')) {
                    if (!name.empty()) {
                        preferred_names.push_back(name);
                    }
                }
            }
            if (!preferred_names.empty()) {
                std::vector<ibv_device*> ordered;
                for (const auto& preferred_name : preferred_names) {
                    auto it = std::find_if(
                        ib_devs.begin(), ib_devs.end(),
                        [&preferred_name](ibv_device* dev) {
                            const char* name = ibv_get_device_name(dev);
                            return name != nullptr && preferred_name == name;
                        });
                    if (it == ib_devs.end()) {
                        std::string available;
                        for (size_t i = 0; i < ib_devs.size(); ++i) {
                            if (i > 0) available += ", ";
                            available += ibv_get_device_name(ib_devs[i]);
                        }
                        ibv_free_device_list(dev_list);
                        throw std::runtime_error(
                            "Configured NIC not available for rank " + std::to_string(rank) +
                            ": " + preferred_name + ", active candidates: " +
                            (available.empty() ? "none" : available));
                    }
                    ordered.push_back(*it);
                }
                ibv_free_device_list(dev_list);
                return ordered;
            }
        }

        std::sort(ib_devs.begin(), ib_devs.end(),
                  [](ibv_device* a, ibv_device* b) {
                      return parse_nic_index(ibv_get_device_name(a)) < parse_nic_index(ibv_get_device_name(b));
                  });
        ibv_free_device_list(dev_list);
        return ib_devs;
    }

    void rdma_handler::Init_hca(HCA &hca, ibv_device* device, int device_id, int threadIdx) 
    {
        cudaSetDevice(device_id);
        hca.name = ibv_get_device_name(device);
        hca.ctx = ibv_open_device(device);
        if (!hca.ctx) {
            throw std::runtime_error("Failed to open device: " + hca.name);
        }

        hca.pd = ibv_alloc_pd(hca.ctx);
        if (!hca.pd) {
            ibv_close_device(hca.ctx);
            throw std::runtime_error("Failed to allocate protection domain for device: " + hca.name);
        }

        hca.cq = ibv_create_cq(hca.ctx, 10, nullptr, nullptr, 0);
        if (!hca.cq) {
            ibv_dealloc_pd(hca.pd);
            ibv_close_device(hca.ctx);
            throw std::runtime_error("Failed to create completion queue for device: " + hca.name);
        }

        ibv_qp_init_attr qp_init_attr = {};
        qp_init_attr.send_cq = hca.cq;
        qp_init_attr.recv_cq = hca.cq;
        qp_init_attr.qp_type = IBV_QPT_RC;
        qp_init_attr.cap.max_send_wr = 32;
        qp_init_attr.cap.max_recv_wr = 32;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;

        hca.qp = ibv_create_qp(hca.pd, &qp_init_attr);
        if (!hca.qp) {
            ibv_destroy_cq(hca.cq);
            ibv_dealloc_pd(hca.pd);
            ibv_close_device(hca.ctx);
            throw std::runtime_error("Failed to create queue pair for device: " + hca.name);
        }

        cudaMalloc(&hca.recv_signal, sizeof(uint32_t));
        cudaMemset(hca.recv_signal, 0, sizeof(uint32_t));
        hca.recv_signal_mr = ibv_reg_mr(
            hca.pd,
            hca.recv_signal, 
            sizeof(uint32_t),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
        if (!hca.recv_signal_mr) {
            ibv_destroy_qp(hca.qp);
            ibv_destroy_cq(hca.cq);
            ibv_dealloc_pd(hca.pd);
            ibv_close_device(hca.ctx);
            throw std::runtime_error("Failed to register MR for recv_signal on device: " + hca.name);
        }

        hca.send_signal = (uint32_t*)malloc(sizeof(uint32_t));
        memset(hca.send_signal, 0, sizeof(uint32_t));
        hca.send_signal_mr = ibv_reg_mr(
            hca.pd,
            hca.send_signal, 
            sizeof(uint32_t),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE );
        if (!hca.send_signal_mr) {
            ibv_dereg_mr(hca.recv_signal_mr);
            cudaFree(hca.recv_signal);
            ibv_destroy_qp(hca.qp);
            ibv_destroy_cq(hca.cq);
            ibv_dealloc_pd(hca.pd);
            ibv_close_device(hca.ctx);
            throw std::runtime_error("Failed to register MR for send_signal on device: " + hca.name);
        }

        for (int i = 0; i < BufferCount; ++i) {
            ibv_mr* send_mr = ibv_reg_mr(
            hca.pd,
            Send_Buffers[i + threadIdx * BufferCount]->buffer, 
            BufferSize,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
            if (!send_mr) {
                throw std::runtime_error("Failed to register MR for Send_Buffer " + std::to_string(i));
            }
            hca.send_mrs.push_back(send_mr);

            ibv_mr* recv_mr = ibv_reg_mr(
                hca.pd,
                Recv_Buffers[i + threadIdx * BufferCount]->buffer, 
                BufferSize,
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE
            );
            if (!recv_mr) {
                throw std::runtime_error("Failed to register MR for Recv_Buffer " + std::to_string(i));
            }
            hca.recv_mrs.push_back(recv_mr);
        }    
    }

    void rdma_handler::setWorkerId(int thread_id)
    {
        if (threadIndex_Map.find(std::this_thread::get_id()) != threadIndex_Map.end()) {
            throw std::runtime_error("Thread ID already set for this thread");
        }
        if (thread_id < 0 || thread_id >= worker_num_) {
            throw std::out_of_range("Invalid thread ID");
        }
        std::thread::id this_id = std::this_thread::get_id();
        threadIndex_Map[this_id] = thread_id;
    }

    int rdma_handler::getWorkerIndex()
    {
        std::thread::id this_id = std::this_thread::get_id();
        return threadIndex_Map[this_id];
    }

    void rdma_handler::setup_connection(std::string ip_port, int device_id, int rank, int worker_num) 
    {
        worker_num_ = worker_num;
        baseIpPort = ip_port;
        cudaSetDevice(device_id);
        sockfd = socket_connect(ip_port, rank);
        threadSockt_Map[std::this_thread::get_id()] = sockfd;
        Logger::get_instance().log(LOG_INFO, "Socket connection established to %s", ip_port.c_str());
        int bandwidth = 200; // in Gbps
        std::vector<ibv_device*> dev_list = Select_IB_Devices(bandwidth, rank);
        if (dev_list.empty()) {
            throw std::runtime_error("No suitable RDMA device found");
        }
        if (dev_list.size() < 2) {
            throw std::runtime_error("At least 2 RDMA devices are required by current backend implementation");
        }
        if (dev_list.size() > 2) {
            Logger::get_instance().log(
                LOG_WARN,
                "Detected %zu RDMA devices, but current backend supports 2 NICs. Truncating to the first 2 devices.",
                dev_list.size());
            dev_list.resize(2);
        }
        NIC_num_ = dev_list.size();
        send_signal_seq_.assign(static_cast<size_t>(worker_num_) * static_cast<size_t>(NIC_num_), 1u);
        recv_signal_expected_.assign(static_cast<size_t>(worker_num_) * static_cast<size_t>(NIC_num_), 1u);
        Logger::get_instance().log(LOG_INFO, "Selected %zu RDMA devices: %s, %s", dev_list.size(), ibv_get_device_name(dev_list[0]), dev_list.size() > 1 ? ibv_get_device_name(dev_list[1]) : "N/A");

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        const size_t required_bytes = static_cast<size_t>(2) * static_cast<size_t>(BufferCount) *
                                      static_cast<size_t>(worker_num) * static_cast<size_t>(BufferSize);
        Logger::get_instance().log(
            LOG_INFO,
            "RDMA buffer allocation plan: BufferSize=%.3f GiB, BufferCount=%d, workers=%d, required=%.3f GiB, free=%.3f GiB, total=%.3f GiB",
            static_cast<double>(BufferSize) / static_cast<double>(1ULL << 30),
            BufferCount,
            worker_num,
            static_cast<double>(required_bytes) / static_cast<double>(1ULL << 30),
            static_cast<double>(free_bytes) / static_cast<double>(1ULL << 30),
            static_cast<double>(total_bytes) / static_cast<double>(1ULL << 30));

        if (required_bytes > free_bytes) {
            std::ostringstream oss;
            oss << "Insufficient free GPU memory for RDMA buffers. required="
                << static_cast<double>(required_bytes) / static_cast<double>(1ULL << 30)
                << " GiB, free="
                << static_cast<double>(free_bytes) / static_cast<double>(1ULL << 30)
                << " GiB, BufferSize="
                << static_cast<double>(BufferSize) / static_cast<double>(1ULL << 30)
                << " GiB, BufferCount=" << BufferCount
                << ", workers=" << worker_num
                << ". Reduce --worker_num or BufferSize.";
            throw std::runtime_error(oss.str());
        }

        for (int i = 0; i < BufferCount * worker_num; ++i) {
            RDMABlock* buffer = new RDMABlock();
            cudaError_t err = cudaMalloc(&buffer->buffer, BufferSize);
            if (err != cudaSuccess) {
                std::ostringstream oss;
                oss << "Failed to allocate Send_Buffers[" << i << "] for RDMA: "
                    << cudaGetErrorString(err)
                    << " (BufferSize="
                    << static_cast<double>(BufferSize) / static_cast<double>(1ULL << 30)
                    << " GiB)";
                throw std::runtime_error(oss.str());
            }
            err = cudaMalloc(&buffer->flag, sizeof(bool));
            if (err != cudaSuccess) {
                std::ostringstream oss;
                oss << "Failed to allocate Send_Buffers[" << i << "].flag: "
                    << cudaGetErrorString(err);
                throw std::runtime_error(oss.str());
            }
            buffer->size = BufferSize;
            Send_Buffers.push_back(buffer);
        }
        for (int i = 0; i < BufferCount * worker_num; ++i) {
            RDMABlock* buffer = new RDMABlock();
            cudaError_t err = cudaMalloc(&buffer->buffer, BufferSize);
            if (err != cudaSuccess) {
                std::ostringstream oss;
                oss << "Failed to allocate Recv_Buffers[" << i << "] for RDMA: "
                    << cudaGetErrorString(err)
                    << " (BufferSize="
                    << static_cast<double>(BufferSize) / static_cast<double>(1ULL << 30)
                    << " GiB)";
                throw std::runtime_error(oss.str());
            }
            err = cudaMalloc(&buffer->flag, sizeof(bool));
            if (err != cudaSuccess) {
                std::ostringstream oss;
                oss << "Failed to allocate Recv_Buffers[" << i << "].flag: "
                    << cudaGetErrorString(err);
                throw std::runtime_error(oss.str());
            }
            buffer->size = BufferSize;
            Recv_Buffers.push_back(buffer);
        }
        for (int i = 0; i < worker_num; i++)
        {
            for (const auto& dev : dev_list) {
                hcas.push_back(HCA());
                Init_hca(hcas.back(), dev, device_id, i);
            }
        }

        local_qp_infos.resize(hcas.size());
        remote_qp_infos.resize(hcas.size());
        for (int i = 0; i < hcas.size(); ++i) {
            ibv_qp_attr qp_attr = {};
            qp_attr.qp_state = IBV_QPS_INIT;
            qp_attr.port_num = hcas[i].port;
            qp_attr.pkey_index = 0;
            qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
            if (ibv_modify_qp(hcas[i].qp, &qp_attr,
                            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {    
                throw std::runtime_error("Failed to modify QP to INIT state for device: " + hcas[i].name);
                            }   
            local_qp_infos[i].qpn = hcas[i].qp->qp_num;
            local_qp_infos[i].psn = lrand48() & 0xffffff;
            for (int j = 0; j < BufferCount; ++j) {
                local_qp_infos[i].send_rkeys[j] = hcas[i].send_mrs[j]->rkey;
                local_qp_infos[i].send_vaddrs[j] = reinterpret_cast<uint64_t>(hcas[i].send_mrs[j]->addr);
                local_qp_infos[i].recv_rkeys[j] = hcas[i].recv_mrs[j]->rkey;
                local_qp_infos[i].recv_vaddrs[j] = reinterpret_cast<uint64_t>(hcas[i].recv_mrs[j]->addr);
            }
            local_qp_infos[i].recv_signal_rkey = hcas[i].recv_signal_mr->rkey;  
            local_qp_infos[i].recv_signal_vaddr = reinterpret_cast<uint64_t>(hcas[i].recv_signal);
            struct ibv_port_attr port_attr;
            if (ibv_query_port(hcas[i].ctx, hcas[i].port, &port_attr)) {
                throw std::runtime_error("Failed to query port for device: " + hcas[i].name);
            }
            local_qp_infos[i].lid = port_attr.lid;
        }
        const size_t qp_info_size = sizeof(local_qp_infos[0]);
        if (rank == 0)
        {
            // 使用正确的大小进行写入和读取
            write(sockfd, local_qp_infos.data(), qp_info_size * hcas.size());
            read(sockfd, remote_qp_infos.data(), qp_info_size * hcas.size());
        } 
        else if (rank == 1)
        {
            // 使用正确的大小进行写入和读取
            read(sockfd, remote_qp_infos.data(), qp_info_size * hcas.size());
            write(sockfd, local_qp_infos.data(), qp_info_size * hcas.size());
        }
        Logger::get_instance().log(LOG_INFO, "QP info exchange complete for rank %d", rank);
        for (int i = 0; i < hcas.size(); ++i) {
            ibv_qp_attr qp_attr = {};
            qp_attr.qp_state = IBV_QPS_RTR;
            qp_attr.path_mtu = IBV_MTU_4096;
            qp_attr.dest_qp_num = remote_qp_infos[i].qpn;
            qp_attr.rq_psn = remote_qp_infos[i].psn;

            qp_attr.max_dest_rd_atomic = 1;
            qp_attr.min_rnr_timer = 12;
            qp_attr.ah_attr.is_global = 0;
            qp_attr.ah_attr.dlid = remote_qp_infos[i].lid;
            qp_attr.ah_attr.sl = 0;
            qp_attr.ah_attr.src_path_bits = 0;
            qp_attr.ah_attr.port_num = hcas[i].port;

            if (ibv_modify_qp(hcas[i].qp, &qp_attr,
                            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                            IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                            IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
                throw std::runtime_error("Failed to modify QP to RTR state for device: " + hcas[i].name);
            }

            memset(&qp_attr, 0, sizeof(qp_attr));
            qp_attr.qp_state = IBV_QPS_RTS;
            qp_attr.timeout = 14;
            qp_attr.retry_cnt = 7;
            qp_attr.rnr_retry = 7; // infinite retry
            qp_attr.sq_psn = local_qp_infos[i].psn;
            qp_attr.max_rd_atomic = 1;

            if (ibv_modify_qp(hcas[i].qp, &qp_attr,
                            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
                throw std::runtime_error("Failed to modify QP to RTS state for device: " + hcas[i].name);
            }
        }
        Logger::get_instance().log(LOG_INFO, "QP state transitions complete for rank %d", rank);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::get_instance().log(LOG_ERROR, "CUDA error during RDMA setup connection: %s", cudaGetErrorString(err));
            throw std::runtime_error("CUDA error during RDMA setup connection");
        }
        barrier(rank);
    }

    bool poll_one_wc(struct ibv_cq* cq, int timeout_us) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(timeout_us);
        while (std::chrono::steady_clock::now() < deadline) {
            ibv_wc wc;
            int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0) {
                throw std::runtime_error("Failed to poll CQ");
            }
            if (n == 0) {
                continue;
            }
            if (wc.status != IBV_WC_SUCCESS) {
                const char* status_str = ibv_wc_status_str(wc.status);
                fprintf(stderr,
                        "Completion error: status=%d(%s) wr_id=%llu opcode=%d qp=%u\n",
                        wc.status,
                        status_str ? status_str : "unknown",
                        (unsigned long long)wc.wr_id,
                        wc.opcode,
                        wc.qp_num);
                exit(EXIT_FAILURE);
            }
            return true;
        }
        return false;
    }
    
    void rdma_handler::test_connection(int device_id, int rank) {
        cudaSetDevice(device_id);
        if (rank == 0)
        {
            void* test_buffer[BufferCount * worker_num_];
            for (int i = 0; i < BufferCount * worker_num_; ++i)
            {
                cudaMallocHost(&test_buffer[i], sizeof(float) * 8);
                float* test_data = static_cast<float*>(test_buffer[i]);
                for (int j = 0; j < 8; ++j) {
                    test_data[j] = static_cast<float>(j * BufferCount * i + 1);
                }
                cudaMemcpyAsync(Send_Buffers[i]->buffer, test_data, sizeof(float) * 8, cudaMemcpyHostToDevice);
                cudaFreeHost(test_buffer[i]);
            }
            for (int threadIdx = 0; threadIdx < worker_num_; threadIdx++)
            {
                for (int i = 0; i < BufferCount; i++)
                {
                    int N = NIC_num_;
                    const size_t seg_size = (sizeof(float) * 8 + N - 1) / N;

                    // 3. 并行提交 RDMA WRITE 请求
                    //    这个循环会向所有 HCA 快速地提交任务，而不会等待它们完成，从而实现并行。
                    size_t bytes_to_post = sizeof(float) * 8;
                    for (int j = 0; j < N; ++j) {
                        if (bytes_to_post == 0) continue; // 如果缓冲区已经分配完毕，则无需再提交

                        // 计算当前块的起始地址和大小
                        const size_t offset = j * seg_size;
                        const uint64_t local_chunk_addr = local_qp_infos[j + threadIdx * NIC_num_].send_vaddrs[i] + offset;
                        const uint64_t remote_chunk_addr = remote_qp_infos[j + threadIdx * NIC_num_].recv_vaddrs[i] + offset;
                        const size_t current_chunk_size = std::min(seg_size, bytes_to_post);

                        // 准备 Scatter/Gather Entry (SGE)，描述本地内存块
                        ibv_sge sge{};
                        sge.addr = (uintptr_t)local_chunk_addr;
                        sge.length = current_chunk_size;
                        sge.lkey = hcas[j + threadIdx * NIC_num_].send_mrs[i]->lkey;

                        // 准备 Send Work Request (WR)，描述整个 RDMA WRITE 操作
                        ibv_send_wr wr{};
                        wr.wr_id = (uint64_t)(i * N + j); // 创建一个唯一的ID用于调试
                        wr.opcode = IBV_WR_RDMA_WRITE;
                        wr.send_flags = 0; 
                        wr.sg_list = &sge;
                        wr.num_sge = 1;
                        wr.wr.rdma.remote_addr = remote_chunk_addr;
                        wr.wr.rdma.rkey = remote_qp_infos[j + threadIdx * NIC_num_].recv_rkeys[i];

                        // 向第 j 个 HCA 的 QP 提交工作请求
                        ibv_send_wr* bad_wr = nullptr;
                        if (ibv_post_send(hcas[j + threadIdx * NIC_num_].qp, &wr, &bad_wr)) {
                            fprintf(stderr, "Failed to post RDMA READ for hca %d, buffer %d\n", j, i);
                            exit(1); // or throw exception
                        }
                        bytes_to_post -= current_chunk_size;

                        hcas[j + threadIdx * NIC_num_].send_signal[0] = 1;
                        ibv_send_wr signal_wr{};
                        ibv_sge signal_sge{};
                        signal_sge.addr = (uintptr_t)hcas[j + threadIdx * NIC_num_].send_signal;
                        signal_sge.length = sizeof(uint32_t); 
                        signal_sge.lkey = hcas[j + threadIdx * NIC_num_].send_signal_mr->lkey;

                        signal_wr.wr_id = (uint64_t)(i * N + j) + 1000000; 
                        signal_wr.opcode = IBV_WR_RDMA_WRITE; 
                        signal_wr.send_flags = IBV_SEND_SIGNALED;
                        signal_wr.sg_list = &signal_sge;
                        signal_wr.num_sge = 1;
                        signal_wr.wr.rdma.remote_addr = remote_qp_infos[j + threadIdx * NIC_num_].recv_signal_vaddr;
                        signal_wr.wr.rdma.rkey = remote_qp_infos[j + threadIdx * NIC_num_].recv_signal_rkey;
                        
                        ibv_send_wr* bad_signal_wr = nullptr;
                        if (ibv_post_send(hcas[j + threadIdx * NIC_num_].qp, &signal_wr, &bad_signal_wr)) {
                            fprintf(stderr, "Failed to post SIGNAL for hca %d, buffer %d\n", j, i);
                            exit(1);
                        }
                    }
                    barrier(rank);
                }
            }
        }    
        else if (rank == 1)
        {
            for (int threadIdx = 0; threadIdx < worker_num_; threadIdx++)
            {
                for (int i = 0; i < BufferCount; i++)
                {
                    float* recv_data_host;
                    cudaMallocHost(&recv_data_host, sizeof(float) * 8);
                    memset(recv_data_host, 0, sizeof(float)*8);
                    int N = NIC_num_;
                    RDMA_PollingKernel<<<N, 1>>>(hcas[0 + threadIdx * NIC_num_].recv_signal, hcas[1 + threadIdx * NIC_num_].recv_signal);
                    cudaMemcpyAsync(recv_data_host, Recv_Buffers[i + threadIdx * BufferCount]->buffer, sizeof(float) * 8, cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();
                    float* recv_data = static_cast<float*>(recv_data_host);
                    for (int j = 0; j < 8; j++) {
                        if (recv_data[j] != static_cast<float>(j * BufferCount * (threadIdx * BufferCount + i) + 1)) {
                            
                            Logger::get_instance().log(LOG_ERROR, "Data mismatch in RDMA test at buffer %d, index %d", i, j);
                            Logger::get_instance().log(LOG_ERROR, "Expected: %f, Received: %f", static_cast<float>(j * BufferCount + 1), recv_data[j]);
                            throw std::runtime_error("Data mismatch during RDMA connection test");
                        }
                    }
                    cudaFreeHost(recv_data_host);
                    barrier(rank);
                }
            }
        }  
        Logger::get_instance().log(LOG_INFO, "RDMA Connection test successful for device %d, rank %d", device_id, rank);
    }   

    void rdma_handler::getRecvBuffers(void** buffers, int buffer_num, cudaStream_t stream) {
        if (buffer_num <= 0 || buffer_num > BufferCount) {
            throw std::out_of_range("Invalid buffer number");
        }
        const int index = getWorkerIndex();
        const int base = index * NIC_num_;
        if (NIC_num_ == 1) {
            const uint32_t expected0 = recv_signal_expected_[base];
            RDMA_PollingKernelSeq1<<<1, 1, 0, stream>>>(hcas[base].recv_signal, expected0);
            recv_signal_expected_[base] = expected0 + 1;
        } else if (NIC_num_ == 2) {
            const uint32_t expected0 = recv_signal_expected_[base];
            const uint32_t expected1 = recv_signal_expected_[base + 1];
            RDMA_PollingKernelSeq2<<<2, 1, 0, stream>>>(
                hcas[base].recv_signal,
                hcas[base + 1].recv_signal,
                expected0,
                expected1);
            recv_signal_expected_[base] = expected0 + 1;
            recv_signal_expected_[base + 1] = expected1 + 1;
        } else {
            throw std::runtime_error("RDMA polling currently supports up to 2 NICs");
        }
        for (int i = 0; i < buffer_num; ++i)
            *reinterpret_cast<void**>(buffers[i]) = Recv_Buffers[i + index * BufferCount]->buffer;

    }

    void rdma_handler::getSendBuffers(void** buffers, int buffer_num) {
        if (buffer_num <= 0 || buffer_num > BufferCount) {
            throw std::out_of_range("Invalid buffer ID");
        }
        int index = getWorkerIndex();
        for (int i = 0; i < buffer_num; i++)
        {
            *reinterpret_cast<void**>(buffers[i]) = Send_Buffers[i + index * BufferCount]->buffer;
        }
    }

    double rdma_handler::communicate_test(void* pointer, int64_t size, int rank, int device_id) {
        cudaSetDevice(device_id);
        if (pointer == nullptr || size <= 0 || size > static_cast<int64_t>(BufferSize)) {
            return 0.0;
        }
        int index = 0;
        auto it = threadIndex_Map.find(std::this_thread::get_id());
        if (it != threadIndex_Map.end()) {
            index = it->second;
        }
        const int base = index * NIC_num_;
        const int profile_buffer_id = (BufferCount > 1) ? (BufferCount - 1) : 0;
        cudaMemcpy(
            Send_Buffers[profile_buffer_id + index * BufferCount]->buffer,
            pointer,
            size,
            cudaMemcpyDeviceToDevice);
        if (rank == 0)
        {
            constexpr int kCommTestPollTimeoutUs = 2 * 1000 * 1000;
            std::vector<double> latencies;
            for (int j = 0; j < NIC_num_; ++j) {
                drain_cq(hcas[base + j].cq);
            }
            for (int i = 0; i < 20; i++)
            {
                auto start = std::chrono::high_resolution_clock::now();
                int N = NIC_num_;
                const size_t seg_size = (size + N - 1) / N;
                size_t bytes_to_post = size;
                std::vector<bool> posted(N, false);
                for (int j = 0; j < N; ++j) {
                    if (bytes_to_post == 0) continue;
                    const size_t offset = j * seg_size;
                    const uint64_t local_chunk_addr = local_qp_infos[base + j].send_vaddrs[profile_buffer_id] + offset;
                    const uint64_t remote_chunk_addr = remote_qp_infos[base + j].recv_vaddrs[profile_buffer_id] + offset;
                    const size_t current_chunk_size = std::min(seg_size, bytes_to_post);
                    ibv_sge sge{};
                    sge.addr = (uintptr_t)local_chunk_addr;
                    sge.length = current_chunk_size;
                    sge.lkey = hcas[base + j].send_mrs[profile_buffer_id]->lkey;
                    ibv_send_wr wr{};
                    wr.wr_id = (uint64_t(0) * N + j);
                    wr.opcode = IBV_WR_RDMA_WRITE;
                    wr.send_flags = IBV_SEND_SIGNALED;
                    wr.sg_list = &sge;
                    wr.num_sge = 1;
                    wr.wr.rdma.remote_addr = remote_chunk_addr;
                    wr.wr.rdma.rkey = remote_qp_infos[base + j].recv_rkeys[profile_buffer_id];
                    ibv_send_wr* bad_signal_wr = nullptr;
                    int ret = ibv_post_send(hcas[base + j].qp, &wr, &bad_signal_wr);
                    if (ret) {
                        Logger::get_instance().log(
                            LOG_ERROR,
                            "Failed to post RDMA WRITE in communicate_test for hca %d, iter %d, ret=%d, errno=%s",
                            base + j,
                            i,
                            ret,
                            std::strerror(ret));
                        throw std::runtime_error("Failed to post RDMA WRITE in communicate_test");
                    }
                    posted[j] = true;
                    bytes_to_post -= current_chunk_size;
                }
                for (int j = 0; j < N; ++j) {
                    if (posted[j]) {
                        if (!poll_one_wc(hcas[base + j].cq, kCommTestPollTimeoutUs)) {
                            Logger::get_instance().log(
                                LOG_WARN,
                                "communicate_test CQ poll timed out on hca %d (iter=%d, size=%lld); skipping remaining samples",
                                base + j,
                                i,
                                static_cast<long long>(size));
                            return 0.0;
                        }
                    }
                }
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::micro> elapsed = end - start;
                latencies.push_back(elapsed.count());
            }
            if (latencies.empty()) return 0.0;

            size_t n = latencies.size();
            double average = 0.0;

            if (n > 10) {
                std::sort(latencies.begin(), latencies.end());
                double sum = std::accumulate(latencies.begin() + 5, latencies.end() - 5, 0.0);
                average = sum / static_cast<double>(n - 10);
            } else {
                double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
                average = sum / static_cast<double>(n);
            }
            return average;
        }
        return 0.0;
    }

    void rdma_handler::transferBufferRDMA(void** buffers, int buffer_num, int* buffer_size, int device_id, cudaStream_t stream) {
        
        cudaSetDevice(device_id);
        if (buffer_num <= 0 || buffer_num > BufferCount) {
            throw std::out_of_range("Invalid buffer ID");
        }
        const int index = getWorkerIndex();
        const int N = NIC_num_;
        const int base = index * NIC_num_;
        Logger::get_instance().log(LOG_INFO, "Starting RDMA transfer on worker %d with %d buffers", index, buffer_num);
        ibv_send_wr wrs[buffer_num][N];
        for (int i = 0; i < buffer_num; i++)
            for (int j = 0; j < N; j++)
                wrs[i][j] = {};
        ibv_send_wr* bad_wr[buffer_num][N] = {{nullptr}};
        ibv_sge sges[buffer_num][N] = {{{}}};
        ibv_send_wr* bad_signal_wrs[N] = {nullptr};
        ibv_send_wr signal_wrs[N] = {{}};
        ibv_sge signal_sges[N] = {{}};
        for (int i = 0; i < buffer_num; i++)
        {
            int64_t size = buffer_size[i];
            if (size <= 0 || size > BufferSize) {
                throw std::invalid_argument("Invalid size for buffer transfer");
            }
            int buffer_id = i;
            const size_t seg_size = (size + N - 1) / N;
            size_t bytes_to_post = size;
            for (int j = 0; j < N; ++j) {
                if (bytes_to_post == 0) continue; 

                const size_t offset = j * seg_size;
                const uint64_t local_chunk_addr = local_qp_infos[base + j].send_vaddrs[buffer_id] + offset;
                const uint64_t remote_chunk_addr = remote_qp_infos[base + j].recv_vaddrs[buffer_id] + offset;
                const size_t current_chunk_size = std::min(seg_size, bytes_to_post);
                sges[i][j].addr = (uintptr_t)local_chunk_addr;
                sges[i][j].length = current_chunk_size;
                sges[i][j].lkey = hcas[base + j].send_mrs[buffer_id]->lkey;

                wrs[i][j].wr_id = (uint64_t)(buffer_id * N + j); 
                wrs[i][j].opcode = IBV_WR_RDMA_WRITE;
                wrs[i][j].send_flags = IBV_SEND_SIGNALED; 
                wrs[i][j].sg_list = &sges[i][j];
                wrs[i][j].num_sge = 1;
                wrs[i][j].wr.rdma.remote_addr = remote_chunk_addr;
                wrs[i][j].wr.rdma.rkey = remote_qp_infos[base + j].recv_rkeys[buffer_id];
                bytes_to_post -= current_chunk_size;
            }
        }
        for (int j = 0; j < N; j++)
        {
            const int hca_idx = base + j;
            uint32_t seq = 1u;
            const size_t seq_idx = static_cast<size_t>(hca_idx);
            if (seq_idx < send_signal_seq_.size()) {
                seq = send_signal_seq_[seq_idx]++;
            }
            hcas[hca_idx].send_signal[0] = seq;
            signal_sges[j].addr = (uintptr_t)hcas[hca_idx].send_signal;
            signal_sges[j].length = sizeof(uint32_t); 
            signal_sges[j].lkey = hcas[hca_idx].send_signal_mr->lkey;

            signal_wrs[j].wr_id = (uint64_t)(N + j) + 1000000; 
            signal_wrs[j].opcode = IBV_WR_RDMA_WRITE; 
            signal_wrs[j].send_flags = IBV_SEND_SIGNALED;
            signal_wrs[j].sg_list = &signal_sges[j];
            signal_wrs[j].num_sge = 1;
            signal_wrs[j].wr.rdma.remote_addr = remote_qp_infos[hca_idx].recv_signal_vaddr;
            signal_wrs[j].wr.rdma.rkey = remote_qp_infos[hca_idx].recv_signal_rkey;
        }
        bool had_pending_wc = false;
        do {
            had_pending_wc = false;
            for (int j = 0; j < N; ++j) {
                if (drain_cq(hcas[base + j].cq) != 0) {
                    had_pending_wc = true;
                }
            }
        } while (had_pending_wc);

        cudaStreamSynchronize(stream);
        for (int i = 0; i < buffer_num; i++)
        {
            int buffer_id = i;
            for (int j = 0; j < N; j++)
            {
                if (wrs[i][j].num_sge == 0) {
                    continue;
                }
                int ret = ibv_post_send(hcas[base + j].qp, &wrs[i][j], &bad_wr[i][j]);
                if (ret) {
                    Logger::get_instance().log(
                        LOG_ERROR,
                        "Failed to post RDMA WRITE for hca %d, buffer %d, error is %d",
                        base + j,
                        buffer_id,
                        ret);
                    throw std::runtime_error("Failed to post RDMA WRITE");
                }
            }
        }
        for (int j = 0; j < N; j++)
        {
            if (ibv_post_send(hcas[base + j].qp, &signal_wrs[j], &bad_signal_wrs[j])) {
                Logger::get_instance().log(LOG_ERROR, "Failed to post SIGNAL for hca %d", base + j);
                throw std::runtime_error("Failed to post SIGNAL");
            }
        }

    }

    void rdma_handler::prepareSendBuffers(void** buffers, int buffer_num, int device_id, cudaStream_t stream) 
    {
        cudaSetDevice(device_id);    
        if (buffer_num <= 0 || buffer_num > BufferCount) {
            throw std::out_of_range("Invalid buffer ID");
        }
        int index = getWorkerIndex();
        for (int i = 0; i < buffer_num; i++)
        {
            cudaMemcpyAsync(Send_Buffers[i + index * BufferCount]->buffer, *reinterpret_cast<void**>(buffers[i]), BufferSize, cudaMemcpyDeviceToDevice, stream);
            if (cudaGetLastError() != cudaSuccess) {
                Logger::get_instance().log(LOG_ERROR, "CUDA error during prepareSendBuffers: %s", cudaGetErrorString(cudaGetLastError()));
                throw std::runtime_error("CUDA error during prepareSendBuffers");
            }
        }
    }

    int rdma_handler::drain_cq(struct ibv_cq* cq) 
    {
        const int MAX_COMPLETIONS_PER_POLL = 128; 
        struct ibv_wc wc[MAX_COMPLETIONS_PER_POLL];
        
        int total_drained = 0;
        int completions;

        do {
            completions = ibv_poll_cq(cq, MAX_COMPLETIONS_PER_POLL, wc);
            
            if (completions < 0) {
                throw std::runtime_error("Failed to clear CQ");
            }
            total_drained += completions;

        } while (completions > 0); 

        return total_drained;
    }

    bool rdma_handler::barrier(int rank) {
        std::thread::id this_id = std::this_thread::get_id();
        if (threadSockt_Map.find(this_id) == threadSockt_Map.end()) {
            // Single-worker mode can safely reuse the already established control socket.
            if (worker_num_ == 1) {
                threadSockt_Map[this_id] = sockfd;
            } else {
                std::string newIpPort = baseIpPort;
                size_t colon_pos = newIpPort.find_last_of(':');
                if (colon_pos != std::string::npos) {
                    int port = std::stoi(newIpPort.substr(colon_pos + 1));
                    port += getWorkerIndex();
                    port += 1;
                    newIpPort = newIpPort.substr(0, colon_pos + 1) + std::to_string(port);
                } else {
                    throw std::runtime_error("Invalid IP:Port format");
                }
                int new_sockfd = socket_connect(newIpPort, rank);
                threadSockt_Map[this_id] = new_sockfd;
            }
        }
        int sockfd_ = threadSockt_Map[this_id];
        if (rank == 0)
        {
            bool barrier = true;
            write(sockfd_, &barrier, sizeof(bool));
            bool peer_barrier;
            read(sockfd_, &peer_barrier, sizeof(bool));
            if (!peer_barrier) {
                perror("Peer did not acknowledge barrier");
                return false;
            }
        }
        else if (rank == 1)
        {
            bool peer_barrier;
            read(sockfd_, &peer_barrier, sizeof(bool));
            if (!peer_barrier) {
                perror("Peer did not acknowledge barrier");
                return false;
            }
            bool barrier = true;
            write(sockfd_, &barrier, sizeof(bool));
        }
        return true;
    }

} // namespace FluidGPU
