#ifndef RUNTIME_H
#define RUNTIME_H

#include "utask.h"
#include "logger.h"
#include "comm_backend.h"
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <getopt.h>
#include <cstring>
#include <set>
#include <map>
#include <thread>
#include <mutex>
#include <fstream>


#define Profiling_mode 0
#define Normal_mode 1
#define Disaggregation_mode 2

#define PCIE 0
#define RDMA 1



namespace FluidGPU{

    class global_scheduler
    {
        public:
            std::map<std::thread::id, std::vector<uTask> > taskQueue_Map;
            std::map<std::thread::id, cudaStream_t> asyncStream_Map;
            std::vector<cudaStream_t> stream_list;

            global_scheduler(){}

            ~global_scheduler(){}

            void add_task(uTask& task);
    };

    class runtime
    {
        private:
            struct rank_switch_comm_stat {
                std::string task_name;
                double recv_time_sum_us = 0.0;
                double send_time_sum_us = 0.0;
                int recv_count = 0;
                int send_count = 0;
            };

            int rank;
            int rank_override;
            int runtime_mode;
            int comm_backend_type;
            bool comm_profile_enabled;
            std::map<std::thread::id, int> taskId_Map;
            int device_id;
            int task_num;
            int comm_mode;
            int worker_num;
            std::vector<int> rankList;
            std::string ip_port;
            std::map<unsigned, rank_switch_comm_stat> rank_switch_comm_stats;
            Logger logger;
            global_scheduler scheduler;
            comm_backend* comm_handler;
            static thread_local bool rankChangeSignal;
            static thread_local bool pendingCopySignal;
            std::mutex task_mutex;
            std::mutex stream_mutex;
            std::mutex comm_stats_mutex;

        public:
            runtime()
            {
                Logger::get_instance().log(LOG_INFO, "Runtime initialized");
                rank = 0;
                rank_override = -1;
                runtime_mode = Normal_mode;
                comm_backend_type = RDMA;
                comm_profile_enabled = false;
                device_id = 0;
                task_num = 0;
                comm_mode = ZeroCopy_mode;
                worker_num = 1;
            }
            ~runtime()
            {
                // if (runtime_mode == Profiling_mode) 
                //     generateDAG();
                Logger::get_instance().log(LOG_INFO, "Runtime is being destroyed");
            }

            void init(int argc, char** argvs);
            static runtime& get_instance();
            void set_rank(int r);
            void readLog(const std::string& _log_file = "../Add_kernel/profiling.json");
            void synchronize();
            int get_device_id() const { return device_id; }
            int get_worker_num() const { return worker_num; }
            int get_rank() const;
            void set_runtime_mode(int mode);
            int get_runtime_mode() const;
            bool is_comm_profile_enabled() const;
            void add_task(uTask& task);
            int get_taskid() const;
            int get_First_TaskRank() const;
            int get_Last_TaskRank() const;
            int getTaskRank() const;
            void set_taskid(int id);
            uTask& get_task();
            uTask& lastTask();
            bool rankCheck(uTask& task);
            bool taskListCheck();
            void generateDAG();
            void preLaunchCheck(uTask& task, cudaStream_t stream=NULL);
            void postLaunchCheck(uTask& task, cudaStream_t stream=NULL);
            void profileCommOnly(uTask& task);
            double communicate_test(void* pointer, int64_t size);
            cudaStream_t& getStream();
            void getResult(void* pointer, int64_t size);
            bool barrier();
            void reset_rank_switch_comm_stats();
            void record_rank_switch_recv(const uTask& task, double elapsed_us);
            void record_rank_switch_send(const uTask& task, double elapsed_us);
            bool dump_rank_switch_comm_csv(const std::string& output_csv_path);
            void setWorkerId(int thread_id);
            
    };
}

#endif // RUNTIME_H
