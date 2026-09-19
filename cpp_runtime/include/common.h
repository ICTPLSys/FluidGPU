#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <cstdint>
#include <atomic>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string>
#include <iostream>
#include <unistd.h>
#include <infiniband/verbs.h>
#include <vector>


// Layout of the shared control block
struct ShmBlock {
    cudaIpcMemHandle_t memHandle;
    cudaIpcEventHandle_t eventHandle;
    std::atomic<bool> ready; 
};

struct BufferBlock{
    void* buffer;
    size_t size;
    cudaEvent_t event;
    size_t shm_id;
};

struct HCA {
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* cq = nullptr;
    ibv_qp* qp = nullptr;
    std::vector<ibv_mr*> send_mrs; 
    std::vector<ibv_mr*> recv_mrs; 
    uint32_t* recv_signal;
    ibv_mr* recv_signal_mr;
    uint32_t* send_signal;
    ibv_mr* send_signal_mr;
    std::string name;
    uint8_t port = 1;
    ibv_sge sge;
    ibv_send_wr wr;
};



struct RDMABlock{
    void* buffer;
    bool* flag;
    size_t size;
};

int socket_connect(const std::string& ip_port, int rank);

#endif // COMMON_H
