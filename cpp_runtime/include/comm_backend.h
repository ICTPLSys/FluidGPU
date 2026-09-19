#ifndef COMM_BACKEND_H
#define COMM_BACKEND_H

#include <vector>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#include <thread>
#include <map>
#include "common.h"
#include "logger.h"

#define BufferSize (1024ULL << 19) // 2 GB
#define BufferCount 2

#define ZeroCopy_mode 0
#define OneCopy_mode 1

namespace FluidGPU{
    class comm_backend{
        public:
            virtual void setup_connection(std::string ip_port, int device_id, int rank, int worker_num = 1) = 0;
            virtual void test_connection(int device_id, int rank) = 0;
            virtual void getRecvBuffers(void** buffers, int buffer_num, cudaStream_t stream=NULL) = 0;
            virtual void getSendBuffers(void** buffers, int buffer_num) = 0;
            virtual double communicate_test(void* pointer, int64_t size, int rank, int device_id) = 0;
            virtual void transferBufferPeer(void* pointer, int64_t size, int buffer_id, int device_id, cudaStream_t stream = NULL) = 0;
            virtual void transferBufferRDMA(void** buffers, int buffer_num, int* buffer_size, int device_id, cudaStream_t stream = NULL) {}
            virtual void prepareSendBuffers(void** buffers, int buffer_num, int device_id, cudaStream_t = NULL) = 0;
            virtual void Record(int buffer_id, int device_id, cudaStream_t stream = NULL) = 0;
            virtual bool barrier(int rank) = 0; 
            virtual void setWorkerId(int thread_id) {}
            virtual ~comm_backend() = default;
    };

    class pcie_handler : public comm_backend{
        private:
            std::vector<ShmBlock*> shm_blocks;
            std::vector<BufferBlock*> buffers;
            std::vector<BufferBlock*> peer_recv_buffers;
            int sockfd;

        public:
            pcie_handler() {
                // Initialize PCIe handler if needed
            }
            ~pcie_handler() {
                // Clean up PCIe resources if needed
            }
            void setup_connection(std::string ip_port, int device_id, int rank, int worker_num = 1);  
            void test_connection(int device_id, int rank);
            void getRecvBuffers(void** buffers, int buffer_num, cudaStream_t stream=NULL);
            void getSendBuffers(void** buffers, int buffer_num);
            double communicate_test(void* pointer, int64_t size, int rank, int device_id);
            void transferBufferPeer(void* pointer, int64_t size, int buffer_id, int device_id, cudaStream_t stream = NULL);
            void transferBufferRDMA(void** buffers, int buffer_num, int* buffer_size, int device_id, cudaStream_t stream = NULL) {};
            void prepareSendBuffers(void** buffers, int buffer_num, int device_id, cudaStream_t stream = NULL) {}
            void Record(int buffer_id, int device_id, cudaStream_t stream = NULL);
            bool barrier(int rank);
            void setWorkerId(int thread_id) {}
            void debug_print_buffer_addresses(int device_id, int rank);
    };

    class rdma_handler : public comm_backend{
        private:
            struct qp_info {
                uint32_t qpn;
                uint16_t lid;
                uint32_t psn;
                uint32_t send_rkeys[BufferCount];
                uint64_t send_vaddrs[BufferCount];
                uint32_t recv_rkeys[BufferCount];
                uint64_t recv_vaddrs[BufferCount];
                uint32_t recv_signal_rkey;
                uint64_t recv_signal_vaddr;
            };

        private:
            int sockfd;
            int worker_num_;
            int NIC_num_;
            std::string baseIpPort;
            std::map<std::thread::id, int> threadIndex_Map;
            std::map<std::thread::id, int> threadSockt_Map;
            std::vector<HCA> hcas;
            std::vector<RDMABlock*> Send_Buffers;
            std::vector<RDMABlock*> Recv_Buffers;
            std::vector<uint32_t> send_signal_seq_;
            std::vector<uint32_t> recv_signal_expected_;
            std::vector<qp_info> local_qp_infos;
            std::vector<qp_info> remote_qp_infos;
            std::vector<ibv_device*> Select_IB_Devices(int required_bandwidth_gbps=200, int rank=0);
            void Init_hca(HCA &hca, ibv_device* device, int device_id=0, int threadIdx=0);
            int drain_cq(ibv_cq* cq);
            void setWorkerId(int thread_id);
            int getWorkerIndex();


        public:
            rdma_handler()
            {

            }

            ~rdma_handler()
            {

            }

            void setup_connection(std::string ip_port, int device_id, int rank, int worker_num = 1);  
            void test_connection(int device_id, int rank);
            void getRecvBuffers(void** buffers, int buffer_num, cudaStream_t stream=NULL);
            void getSendBuffers(void** buffers, int buffer_num);
            double communicate_test(void* pointer, int64_t size, int rank, int device_id);
            void transferBufferPeer(void* pointer, int64_t size, int buffer_id, int device_id, cudaStream_t stream = NULL) {};
            void transferBufferRDMA(void** buffers, int buffer_num, int* buffer_size, int device_id, cudaStream_t stream = NULL);
            void prepareSendBuffers(void** buffers, int buffer_num, int device_id, cudaStream_t stream = NULL);
            void Record(int buffer_id, int device_id, cudaStream_t stream = NULL) {};
            bool barrier(int rank);
    };
}

#endif // COMM_BACKEND_H
