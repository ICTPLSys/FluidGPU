#include "runtime.h"
#include <regex>
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <stdexcept>

namespace {
inline bool is_disagg_execution_mode(int runtime_mode, bool comm_profile_enabled) {
    return runtime_mode == Disaggregation_mode ||
           (runtime_mode == Profiling_mode && comm_profile_enabled);
}

inline std::string to_lower_copy(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

[[noreturn]] void print_runtime_usage_and_exit(const char* prog) {
    std::cerr
        << "Usage: " << prog
        << " -r <runtime_mode> -i <ip:port> -c <comm_backend> -g <gpu_id>"
        << " -n <worker_num> -l <log_mode> [--rank <rank>] [--comm-profile]\n";
    std::exit(EXIT_FAILURE);
}

int parse_non_negative_int_or_exit(const char* raw, const char* name) {
    try {
        size_t pos = 0;
        const long long value = std::stoll(raw, &pos);
        if (pos != std::strlen(raw) || value < 0 || value > INT_MAX) {
            throw std::out_of_range("out_of_range");
        }
        return static_cast<int>(value);
    } catch (...) {
        std::cerr << "Invalid " << name << ": " << raw << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
}


void parseCommandLineArguments(int argc, char **argv, int& runtime_mode, int& comm_backend, std::string& ip_port,
                               int& device_id, int& worker_num, int& rank_override, bool& comm_profile_enabled)  {
    int opt;
    int option_index = 0;
    struct option long_options[] = {
        {"runtime_mode", required_argument, 0, 'r'},
        {"runtime-mode", required_argument, 0, 'r'},
        {"ip_port", required_argument, 0, 'i'},
        {"ip-port", required_argument, 0, 'i'},
        {"comm_backends", required_argument, 0, 'c'},
        {"comm_backend", required_argument, 0, 'c'},
        {"comm-backend", required_argument, 0, 'c'},
        {"gpu_id", required_argument, 0, 'g'},
        {"gpu-id", required_argument, 0, 'g'},
        {"logger_mode", required_argument, 0, 'l'},
        {"logger-mode", required_argument, 0, 'l'},
        {"worker_num", required_argument, 0, 'n'},
        {"worker-num", required_argument, 0, 'n'},
        {"batch", required_argument, 0, 'b'},
        {"rank", required_argument, 0, 'k'},
        {"ctrl-port-base", required_argument, 0, 'p'},
        {"ctrl-bind-ip", required_argument, 0, 'h'},
        {"validate-only", no_argument, 0, 'v'},
        {"check-samples", required_argument, 0, 's'},
        {"check-threshold", required_argument, 0, 't'},
        {"comm-profile", no_argument, 0, 'x'},
        {0, 0, 0, 0}
    };
    if (argc < 2) {
        print_runtime_usage_and_exit(argv[0]);
    }

    opterr = 0; // Handle getopt errors in a unified way below.
    optind = 1; // Ensure deterministic behavior if parser is called more than once.
    while ((opt = getopt_long(argc, argv, "r:i:c:g:l:n:b:k:p:h:vs:t:x", long_options, &option_index)) != -1)
    {
        switch (opt) {
            case 'r': {
                const std::string mode = to_lower_copy(optarg);
                if (mode == "profiling") {
                    runtime_mode = Profiling_mode;
                } else if (mode == "normal") {
                    runtime_mode = Normal_mode;
                } else if (mode == "dist" || mode == "disaggregation") {
                    runtime_mode = Disaggregation_mode;
                } else {
                    std::cerr << "Invalid runtime mode: " << optarg << std::endl;
                    print_runtime_usage_and_exit(argv[0]);
                }
                break;
            }
            case 'i':
                ip_port = optarg;
                break;
            case 'c': {
                const std::string backend = to_lower_copy(optarg);
                if (backend == "pcie") {
                    comm_backend = PCIE;
                } else if (backend == "rdma") {
                    comm_backend = RDMA;
                } else {
                    std::cerr << "Invalid communication backend: " << optarg << std::endl;
                    print_runtime_usage_and_exit(argv[0]);
                }
                break;
            }
            case 'g':
                device_id = parse_non_negative_int_or_exit(optarg, "gpu id");
                break;
            case 'l': {
                const std::string log_mode = to_lower_copy(optarg);
                if (log_mode == "debug") {
                    Logger::get_instance().setLogMode(DEBUG);
                } else if (log_mode == "normal") {
                    Logger::get_instance().setLogMode(NORMAL);
                } else {
                    std::cerr << "Invalid log mode: " << optarg << std::endl;
                    print_runtime_usage_and_exit(argv[0]);
                }
                break;
            }
            case 'n':
                worker_num = parse_non_negative_int_or_exit(optarg, "worker number");
                if (worker_num <= 0) {
                    std::cerr << "worker_num must be > 0, got: " << worker_num << std::endl;
                    print_runtime_usage_and_exit(argv[0]);
                }
                break;
            case 'b':
                //batch_size = std::stoi(optarg);
                (void)optarg; // Placeholder to avoid unused variable warning
                break;
            case 'k':
                rank_override = parse_non_negative_int_or_exit(optarg, "rank");
                break;
            case 'p':
            case 'h':
            case 'v':
            case 's':
            case 't':
                // process.cu consumes these options; runtime parser should not reject them.
                break;
            case '?':
                if (optopt != 0) {
                    std::cerr << "Unknown or malformed option: -" << static_cast<char>(optopt) << std::endl;
                } else if (optind > 0 && argv[optind - 1] != nullptr) {
                    std::cerr << "Unknown option: " << argv[optind - 1] << std::endl;
                }
                print_runtime_usage_and_exit(argv[0]);
                break;
            case 'x':
                comm_profile_enabled = true;
                break;
            default:
                print_runtime_usage_and_exit(argv[0]);
        }
    }
}

namespace FluidGPU
{
    thread_local bool runtime::rankChangeSignal = false;
    thread_local bool runtime::pendingCopySignal = false;

    void global_scheduler::add_task(uTask& task)
    {
        std::thread::id this_id = std::this_thread::get_id();
        taskQueue_Map[this_id].push_back(task);
        Logger::get_instance().log(LOG_INFO, "Task with ID %u added to thread %zu's queue", task.task_id, std::thread::id(this_id));
    }

    int allocStream()
    {
        static std::atomic<int> stream_index(0);
        int current_index = stream_index.load();  
        stream_index.fetch_add(1);
        return current_index;
    }

    void runtime::init(int argc, char** argvs)
    {
        parseCommandLineArguments(argc, argvs, runtime_mode, comm_backend_type, ip_port, device_id, worker_num, rank_override, comm_profile_enabled);
        if (comm_profile_enabled && runtime_mode != Profiling_mode) {
            Logger::get_instance().log(LOG_ERROR, "--comm-profile must be used with profiling mode");
            std::cerr << "Error: --comm-profile must be used with '-r profiling'." << std::endl;
            exit(EXIT_FAILURE);
        }
        if (runtime_mode == Normal_mode || runtime_mode == Disaggregation_mode ||
            (runtime_mode == Profiling_mode && comm_profile_enabled))
        {
            readLog("./profiling.json"); // Need to be modified later
        }
        if (rank_override >= 0) {
            rank = rank_override;
            Logger::get_instance().log(LOG_INFO, "Runtime rank overridden to %d by CLI", rank);
        } else if (runtime_mode == Normal_mode) {
            rank = 0;
            Logger::get_instance().log(LOG_INFO, "Runtime rank forced to 0 in normal mode");
        }
        cudaSetDevice(device_id);
        Logger::get_instance().set_rank(rank);
        Logger::get_instance().log(LOG_INFO, "Runtime initialized with mode %d and comm backend %d", runtime_mode, comm_backend_type);
        Logger::get_instance().log(LOG_INFO, "IP Port: %s", ip_port.c_str());
        Logger::get_instance().log(LOG_INFO, "Device ID: %d", device_id);

        std::string runtime_mode_str;
        switch (runtime_mode) {
            case Profiling_mode:
                runtime_mode_str = "Profiling";     
                break;
            case Normal_mode:
                runtime_mode_str = "Normal";
                break;
            case Disaggregation_mode:
                runtime_mode_str = "Disaggregation";
                break;
            default:
                runtime_mode_str = "Unknown";
                break;
        }

        cudaStream_t stream;
        int lowPriority, highPriority;
        cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority);
        // 有几个worker就按照优先级高低创建几个stream
        int avgpriority_num = std::ceil((std::abs(lowPriority - highPriority) + 1) / worker_num);

        for (int i = 0; i < worker_num; i++) {
            int priority = highPriority + (i + 1) / avgpriority_num;
            cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
            scheduler.stream_list.push_back(stream);
        }

        std::string comm_backend_str;
        switch (comm_backend_type)
        {
            case PCIE:
                comm_backend_str = "PCIE";
                break;
            case RDMA:
                comm_backend_str = "RDMA";
            default:
                break;
        }

        std::cout << "Runtime initialized with " << runtime_mode_str << " mode"
                    << ", comm backend " << comm_backend_type
                    << ", IP Port: " << ip_port
                    << ", Device ID: " << device_id
                    << ", Rank: " << rank
                    << ", Comm profile: " << (comm_profile_enabled ? "ON" : "OFF") << std::endl;
                    
        if (comm_backend_type == PCIE) {
            comm_handler = new pcie_handler();
        } else if (comm_backend_type == RDMA) {
            comm_handler = new rdma_handler();
        } else {
            Logger::get_instance().log(LOG_ERROR, "Unknown communication backend type");
            exit(EXIT_FAILURE);
        }
        if (is_disagg_execution_mode(runtime_mode, comm_profile_enabled))
        {
            comm_handler->setup_connection(ip_port, device_id, rank, worker_num);
            comm_handler->test_connection(device_id, rank);
        }
    }

    runtime& runtime::get_instance()
    {
        static runtime instance;
        std::lock_guard<std::mutex> lock(instance.task_mutex);
        std::thread::id this_id = std::this_thread::get_id();
        if (instance.taskId_Map.find(this_id) == instance.taskId_Map.end()) {
            instance.taskId_Map[this_id] = 0;
        }
        if (instance.scheduler.taskQueue_Map.find(this_id) == instance.scheduler.taskQueue_Map.end()) {
            instance.scheduler.taskQueue_Map[this_id] = std::vector<uTask>();
        }
        return instance;
    }

    void runtime::set_rank(int r)
    {
        rank = r;
        Logger::get_instance().log(LOG_INFO, "Runtime rank set to %d", rank);
    }

    void runtime::readLog(const std::string& _log_file)
    {
        std::ifstream file(_log_file);
        Logger::get_instance().log(LOG_INFO, "Read kernel distribution log from %s", _log_file.c_str());
        if (!file.is_open()) {
            std::cout << "Failed to open log file: " << _log_file.c_str() << std::endl;
            exit(1);
        }
        
        nlohmann::json json_data;
        file >> json_data;
        file.close();
        
        task_num = json_data["task_count"];
        rankList.resize(task_num, 0);  // Initialize rankList with size task_num, default to 0
        std::vector<std::string> device_names(task_num);
        for (const auto& task : json_data["tasks"]) {
            int task_id = task["task_id"];
            std::string device_name = task["Device"];
            if (task_id < task_num) {
                device_names[task_id] = device_name;
            } else {
                Logger::get_instance().log(LOG_ERROR, "Task ID %d exceeds task count %d", task_id, task_num);
                exit(EXIT_FAILURE);
            }
        }
        cudaSetDevice(device_id);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device_id);
        std::string current_device_name = deviceProp.name;
        std::cout << "Current device name: " << current_device_name << std::endl;
        std::cout << "first device entry" << device_names[0] << "," << "last device entry" << device_names[task_num - 1] << std::endl;
        if (device_names[0] == current_device_name){
            rank = 0;
        }
        
        // Map unique device names to ranks: the device matching the last entry gets rank 0,
        // all other devices get ranks 1..N-1 in order of first appearance.
        std::string first_device_name = device_names[0];
        std::unordered_map<std::string, int> device_rank_map;
        device_rank_map[first_device_name] = 0;
        int next_rank = 1;

        for(auto& it:device_rank_map){
            Logger::get_instance().log(LOG_INFO, "Device '%s' assigned rank %d", it.first.c_str(), it.second);
        }
        for (const auto& name : device_names) {
            if (name.empty()) continue;
            if (device_rank_map.find(name) == device_rank_map.end()) {
                if (name == first_device_name) continue; // already assigned 0
                device_rank_map[name] = next_rank++;
            }
        }
        for(auto& it:device_rank_map){
            Logger::get_instance().log(LOG_INFO, "Device '%s' assigned rank %d after", it.first.c_str(), it.second);
        }
        // Fill rankList for each task according to mapped device ranks
        for (int i = 0; i < task_num; ++i) {
            const auto& name = device_names[i];
            auto it = device_rank_map.find(name);
            if (it != device_rank_map.end()) {
                rankList[i] = it->second;
            } else {
                // If a device name was missing for some reason, assign the next available rank
                rankList[i] = next_rank;
                device_rank_map[name] = next_rank++;
                Logger::get_instance().log(LOG_INFO, "Assigned new rank %d to previously unseen device '%s' for task %d", rankList[i], name.c_str(), i);
            }
        }

        // Set this runtime's rank based on current device name
        auto it_cur = device_rank_map.find(current_device_name);
        if (it_cur != device_rank_map.end()) {
            rank = it_cur->second;
        } else {
            // If current device not found in the mapping, assign it the next rank
            rank = next_rank;
            device_rank_map[current_device_name] = rank;
            Logger::get_instance().log(LOG_INFO, "Current device '%s' not found in log; assigned rank %d", current_device_name.c_str(), rank);
        }
        
        Logger::get_instance().log(LOG_INFO, "Log file read successfully, task_num: %d", task_num);

    }

    int runtime::get_rank() const
    {
        return rank;
    }

    void runtime::set_runtime_mode(int mode)
    {
        runtime_mode = mode;
        Logger::get_instance().log(LOG_INFO, "Runtime mode set to %d", runtime_mode);
    }

    int runtime::get_runtime_mode() const
    {
        return runtime_mode;
    }

    bool runtime::is_comm_profile_enabled() const
    {
        return comm_profile_enabled;
    }

    void runtime::reset_rank_switch_comm_stats()
    {
        std::lock_guard<std::mutex> lock(comm_stats_mutex);
        rank_switch_comm_stats.clear();
    }

    void runtime::record_rank_switch_recv(const uTask& task, double elapsed_us)
    {
        if (!comm_profile_enabled || elapsed_us <= 0.0) {
            return;
        }
        std::lock_guard<std::mutex> lock(comm_stats_mutex);
        rank_switch_comm_stat& stat = rank_switch_comm_stats[task.task_id];
        stat.task_name = task.task_name;
        stat.recv_time_sum_us += elapsed_us;
        stat.recv_count += 1;
    }

    void runtime::record_rank_switch_send(const uTask& task, double elapsed_us)
    {
        if (!comm_profile_enabled || elapsed_us <= 0.0) {
            return;
        }
        std::lock_guard<std::mutex> lock(comm_stats_mutex);
        rank_switch_comm_stat& stat = rank_switch_comm_stats[task.task_id];
        stat.task_name = task.task_name;
        stat.send_time_sum_us += elapsed_us;
        stat.send_count += 1;
    }

    bool runtime::dump_rank_switch_comm_csv(const std::string& output_csv_path)
    {
        if (!comm_profile_enabled) {
            return false;
        }

        std::lock_guard<std::mutex> lock(comm_stats_mutex);
        if (rank_switch_comm_stats.empty()) {
            return false;
        }

        std::ofstream out(output_csv_path);
        if (!out.is_open()) {
            return false;
        }

        out << "TaskID,TaskName,RecvTimeUsTotal,RecvCount,RecvTimeUsAvg,SendTimeUsTotal,SendCount,SendTimeUsAvg,TotalTimeUs\n";

        double total_recv = 0.0;
        double total_send = 0.0;
        int total_recv_count = 0;
        int total_send_count = 0;

        for (const auto& kv : rank_switch_comm_stats) {
            const unsigned task_id = kv.first;
            const rank_switch_comm_stat& stat = kv.second;
            const double recv_avg = stat.recv_count > 0 ? stat.recv_time_sum_us / static_cast<double>(stat.recv_count) : 0.0;
            const double send_avg = stat.send_count > 0 ? stat.send_time_sum_us / static_cast<double>(stat.send_count) : 0.0;
            const double total_time = stat.recv_time_sum_us + stat.send_time_sum_us;
            out << task_id << ",\"" << stat.task_name << "\","
                << stat.recv_time_sum_us << ","
                << stat.recv_count << ","
                << recv_avg << ","
                << stat.send_time_sum_us << ","
                << stat.send_count << ","
                << send_avg << ","
                << total_time << "\n";

            total_recv += stat.recv_time_sum_us;
            total_send += stat.send_time_sum_us;
            total_recv_count += stat.recv_count;
            total_send_count += stat.send_count;
        }

        const double total_recv_avg = total_recv_count > 0 ? total_recv / static_cast<double>(total_recv_count) : 0.0;
        const double total_send_avg = total_send_count > 0 ? total_send / static_cast<double>(total_send_count) : 0.0;
        out << "__TOTAL__,\"ALL\","
            << total_recv << ","
            << total_recv_count << ","
            << total_recv_avg << ","
            << total_send << ","
            << total_send_count << ","
            << total_send_avg << ","
            << (total_recv + total_send) << "\n";
        return true;
    }

    void runtime::add_task(uTask& task)
    {
        std::thread::id this_id = std::this_thread::get_id();
        taskId_Map[this_id]++;
        scheduler.add_task(task);
    }

    int runtime::get_taskid() const
    {
        std::thread::id this_id = std::this_thread::get_id();
        return taskId_Map.at(this_id);
    }

    int runtime::getTaskRank() const
    {
        if (!is_disagg_execution_mode(runtime_mode, comm_profile_enabled) ||
            rankList.empty())
            return 0;
        const int task_id = get_taskid();
        if (task_id < 0 || static_cast<size_t>(task_id) >= rankList.size()) {
            return 0;
        }
        return rankList[task_id];
    }

    int runtime::get_First_TaskRank() const
    {
        if (!is_disagg_execution_mode(runtime_mode, comm_profile_enabled) ||
            rankList.empty())
            return 0;
        return rankList[0];
    }

    int runtime::get_Last_TaskRank() const
    {
        if (!is_disagg_execution_mode(runtime_mode, comm_profile_enabled) ||
            rankList.empty())
            return 0;
        return rankList.back();
    }

    void runtime::set_taskid(int id)
    {
        std::thread::id this_id = std::this_thread::get_id();
        taskId_Map[this_id] = id;
        Logger::get_instance().log(LOG_INFO, "Task ID set to %d for thread %zu", id, this_id);
    }

    uTask& runtime::get_task()
    {
        std::thread::id this_id = std::this_thread::get_id();
        if (!scheduler.taskQueue_Map[this_id].empty())
        {
            uTask& last_task = scheduler.taskQueue_Map[this_id][taskId_Map[this_id]];
            Logger::get_instance().log(LOG_INFO, "Retrieved task with ID %u for thread %zu", last_task.task_id, this_id);
            taskId_Map[this_id]++;
            return last_task;
        }
    }

    uTask& runtime::lastTask()
    {
        std::thread::id this_id = std::this_thread::get_id();
        if (!scheduler.taskQueue_Map[this_id].empty() && get_taskid() > 0)
        {
            return scheduler.taskQueue_Map[this_id][get_taskid() - 1];
        }
        else
        {
            throw std::runtime_error("No last task available");
        }
    }


    bool runtime::rankCheck(uTask& task)
    {
        if (is_disagg_execution_mode(runtime_mode, comm_profile_enabled))
        {
            if (task.task_rank == rank)
            {
                return true;
            }
            else return false;
        }
        return true; 
    }

    bool runtime::taskListCheck()
    {
        std::thread::id this_id = std::this_thread::get_id();
        if (scheduler.taskQueue_Map[this_id].size() < task_num)
            return true;
        else return false;
    }

    void runtime::generateDAG()
    {
        if (runtime_mode != Profiling_mode)
        {
            Logger::get_instance().log(LOG_INFO, "DAG generation skipped for non-profiling mode");
            return;
        }
        std::thread::id this_id = std::this_thread::get_id();
        taskId_Map[this_id] = 0; 
        int device;
        cudaGetDevice(&device);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::string device_name = deviceProp.name;
        std::ofstream outfile(device_name + "_profiling.csv");
        if (outfile.is_open())
        {
            outfile << "TaskID,TaskName,TaskTime,InputNum,OutputNum\n";
            while (taskId_Map[this_id] < scheduler.taskQueue_Map[this_id].size())
            {
                uTask& task = scheduler.taskQueue_Map[this_id][taskId_Map[this_id]];
                outfile << task.task_id << ","
                        << "\"" << task.task_name << "\"" << ","
                        << task.task_time << ","
                        << task.input_num << ","
                        << task.output_num << "\n";
                taskId_Map[this_id]++;
            }
            outfile.close();
        }
        taskId_Map[this_id] = 0;
        Logger::get_instance().log(LOG_INFO, "DAG generated for device %s", device_name.c_str());
    }

    void runtime::preLaunchCheck(uTask& task, cudaStream_t stream)
    {
        stream = getStream();
        std::thread::id this_id = std::this_thread::get_id();
        if (runtime_mode != Disaggregation_mode)
        {
            Logger::get_instance().log(LOG_INFO, "Pre-launch check skipped for non-disaggregation mode");
            return;
        }
        if (comm_backend_type == PCIE)
        {
            if (task.task_id != 0 && scheduler.taskQueue_Map[this_id].at(task.task_id - 1).task_rank != rank)
            {
                Logger::get_instance().log(LOG_INFO, "Rank change detected for task %u's input buffer", task.task_id);
                if (uTask::bufferTypeCheck(task) == NO_OVERLAP_TASK || uTask::bufferTypeCheck(task) == OVERLAP_TASK)
                {
                    comm_handler->getRecvBuffers(task.input_arrays, task.input_num);
                    if (comm_profile_enabled) {
                        // Use communicate_test-based sampling for comm profiling.
                        double recv_elapsed = 0.0;
                        for (int i = 0; i < task.input_num; ++i) {
                            void* ptr_holder = task.input_arrays ? task.input_arrays[i] : nullptr;
                            const int64_t sz = (task.input_size != nullptr) ? task.input_size[i] : 0;
                            if (ptr_holder == nullptr || sz <= 0) {
                                continue;
                            }
                            void* dev_ptr = *reinterpret_cast<void**>(ptr_holder);
                            if (dev_ptr == nullptr) {
                                continue;
                            }
                            recv_elapsed += communicate_test(dev_ptr, sz);
                        }
                        record_rank_switch_recv(task, recv_elapsed);
                    }
                }
                Logger::get_instance().log(LOG_INFO, "Pre-launch check of input buffer completed for task %u", task.task_id);
            }
            if (task.task_id < scheduler.taskQueue_Map[this_id].size() - 1 && scheduler.taskQueue_Map[this_id].at(task.task_id + 1).task_rank != rank)
            {
                Logger::get_instance().log(LOG_INFO, "Rank change detected for task %u's output buffer", task.task_id);
                rankChangeSignal = true;
                if (uTask::bufferTypeCheck(task) == NO_OVERLAP_TASK && comm_mode == ZeroCopy_mode)
                {
                    comm_handler->getSendBuffers(task.output_arrays, task.output_num);
                }
                else if (uTask::bufferTypeCheck(task) == OVERLAP_TASK || comm_mode == OneCopy_mode)
                {
                    pendingCopySignal = true;
                }
                Logger::get_instance().log(LOG_INFO, "Pre-launch check of output buffer completed for task %u", task.task_id);
            }
        }
        else if (comm_backend_type == RDMA)
        {
            if (task.task_id != 0 && scheduler.taskQueue_Map[this_id].at(task.task_id - 1).task_rank != rank)
            {
                Logger::get_instance().log(LOG_INFO, "Rank change detected for task %u's input buffer", task.task_id);
                if (uTask::bufferTypeCheck(task) == NO_OVERLAP_TASK || uTask::bufferTypeCheck(task) == OVERLAP_TASK)
                {
                    comm_handler->getRecvBuffers(task.input_arrays, task.input_num, stream);
                    if (comm_profile_enabled) {
                        // Use communicate_test-based sampling for comm profiling.
                        double recv_elapsed = 0.0;
                        for (int i = 0; i < task.input_num; ++i) {
                            void* ptr_holder = task.input_arrays ? task.input_arrays[i] : nullptr;
                            const int64_t sz = (task.input_size != nullptr) ? task.input_size[i] : 0;
                            if (ptr_holder == nullptr || sz <= 0) {
                                continue;
                            }
                            void* dev_ptr = *reinterpret_cast<void**>(ptr_holder);
                            if (dev_ptr == nullptr) {
                                continue;
                            }
                            recv_elapsed += communicate_test(dev_ptr, sz);
                        }
                        record_rank_switch_recv(task, recv_elapsed);
                    }
                }
                Logger::get_instance().log(LOG_INFO, "Pre-launch check of input buffer completed for task %u", task.task_id);
            }
            if (task.task_id < scheduler.taskQueue_Map[this_id].size() - 1 && scheduler.taskQueue_Map[this_id].at(task.task_id + 1).task_rank != rank)
            {
                Logger::get_instance().log(LOG_INFO, "Rank change detected for task %u's output buffer", task.task_id);
                rankChangeSignal = true;
                if (uTask::bufferTypeCheck(task) == NO_OVERLAP_TASK)
                {
                    comm_handler->getSendBuffers(task.output_arrays, task.output_num);
                }
                else if (uTask::bufferTypeCheck(task) == OVERLAP_TASK)
                {
                    pendingCopySignal = true;
                }
                Logger::get_instance().log(LOG_INFO, "Pre-launch check of output buffer completed for task %u", task.task_id);
            }
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::get_instance().log(LOG_ERROR, "CUDA error during pre-launch check: %s", cudaGetErrorString(err));
            throw std::runtime_error("CUDA error during pre-launch check");
        }
    }

    void runtime::postLaunchCheck(uTask& task, cudaStream_t stream)
    {
        stream = getStream();
        if (runtime_mode != Disaggregation_mode)
        {
            Logger::get_instance().log(LOG_INFO, "Post-launch check skipped for non-disaggregation mode");
            return;
        }
        if (comm_backend_type == PCIE)
        {
            if (rankChangeSignal == true)
            {
                for (int i = 0; i < task.output_num; i++)
                {
                    if (pendingCopySignal==true)
                    {
                        // Can be optimized by multistreams if the output arrays are more than one
                        comm_handler->transferBufferPeer(task.output_arrays[i], task.output_size[i], i, device_id, stream);
                    }
                    comm_handler->Record(i, device_id, stream); 
                }
                if (comm_profile_enabled) {
                    // Use communicate_test-based sampling for comm profiling.
                    double send_elapsed = 0.0;
                    for (int i = 0; i < task.output_num; ++i) {
                        void* ptr_holder = task.output_arrays ? task.output_arrays[i] : nullptr;
                        const int64_t sz = (task.output_size != nullptr) ? task.output_size[i] : 0;
                        if (ptr_holder == nullptr || sz <= 0) {
                            continue;
                        }
                        void* dev_ptr = *reinterpret_cast<void**>(ptr_holder);
                        if (dev_ptr == nullptr) {
                            continue;
                        }
                        send_elapsed += communicate_test(dev_ptr, sz);
                    }
                    record_rank_switch_send(task, send_elapsed);
                }
                pendingCopySignal = false;
                rankChangeSignal = false;
                Logger::get_instance().log(LOG_INFO, "Post-launch check completed for task %u", task.task_id);
            }
        }
        else if (comm_backend_type == RDMA)
        {
            Logger::get_instance().log(LOG_INFO, "Post-launch check initiated for task %u in worker %zu", task.task_id, std::this_thread::get_id());
            if (rankChangeSignal == true)
            {
                if (pendingCopySignal==true)
                {
                    comm_handler->prepareSendBuffers(task.output_arrays, task.output_num, device_id, stream);
                }
                comm_handler->transferBufferRDMA(task.output_arrays, task.output_num, task.output_size, device_id, stream);
                if (comm_profile_enabled) {
                    // Use communicate_test-based sampling for comm profiling.
                    double send_elapsed = 0.0;
                    for (int i = 0; i < task.output_num; ++i) {
                        void* ptr_holder = task.output_arrays ? task.output_arrays[i] : nullptr;
                        const int64_t sz = (task.output_size != nullptr) ? task.output_size[i] : 0;
                        if (ptr_holder == nullptr || sz <= 0) {
                            continue;
                        }
                        void* dev_ptr = *reinterpret_cast<void**>(ptr_holder);
                        if (dev_ptr == nullptr) {
                            continue;
                        }
                        send_elapsed += communicate_test(dev_ptr, sz);
                    }
                    record_rank_switch_send(task, send_elapsed);
                }
                pendingCopySignal = false;
                rankChangeSignal = false;
                Logger::get_instance().log(LOG_INFO, "Post-launch check completed for task %u in worker %zu", task.task_id, std::this_thread::get_id());
            }
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::get_instance().log(LOG_ERROR, "CUDA error during post-launch check: %s", cudaGetErrorString(err));
            throw std::runtime_error("CUDA error during post-launch check");
        }
    }

    void runtime::profileCommOnly(uTask& task)
    {
        if (!comm_profile_enabled || runtime_mode != Profiling_mode) {
            return;
        }

        std::thread::id this_id = std::this_thread::get_id();
        auto it = scheduler.taskQueue_Map.find(this_id);
        if (it == scheduler.taskQueue_Map.end()) {
            return;
        }
        const std::vector<uTask>& queue = it->second;
        if (queue.empty()) {
            return;
        }

        auto sample_arrays = [this](void** arrays, int* sizes, int num) -> double {
            double elapsed = 0.0;
            for (int i = 0; i < num; ++i) {
                void* ptr_holder = arrays ? arrays[i] : nullptr;
                const int64_t sz = (sizes != nullptr) ? sizes[i] : 0;
                if (ptr_holder == nullptr || sz <= 0) {
                    continue;
                }
                void* dev_ptr = *reinterpret_cast<void**>(ptr_holder);
                if (dev_ptr == nullptr) {
                    continue;
                }
                elapsed += communicate_test(dev_ptr, sz);
            }
            return elapsed;
        };

        const size_t task_idx = static_cast<size_t>(task.task_id);
        if (task_idx > 0 && task_idx < queue.size() && queue[task_idx - 1].task_rank != rank) {
            if (uTask::bufferTypeCheck(task) == NO_OVERLAP_TASK || uTask::bufferTypeCheck(task) == OVERLAP_TASK) {
                const double recv_elapsed = sample_arrays(task.input_arrays, task.input_size, task.input_num);
                record_rank_switch_recv(task, recv_elapsed);
            }
        }

        if (task_idx + 1 < queue.size() && queue[task_idx + 1].task_rank != rank) {
            const double send_elapsed = sample_arrays(task.output_arrays, task.output_size, task.output_num);
            record_rank_switch_send(task, send_elapsed);
        }
    }

    void runtime::getResult(void* pointer, int64_t size)
    {
        if (!is_disagg_execution_mode(runtime_mode, comm_profile_enabled))
        {
            Logger::get_instance().log(LOG_INFO, "Result retrieval skipped for non-disaggregation mode");
            return;
        }
        void** temp_buffers = new void*[1];
        temp_buffers[0] = pointer;
        int temp_buffer_num = 1;
        comm_handler->getRecvBuffers(temp_buffers, temp_buffer_num);
        Logger::get_instance().log(LOG_INFO, "Result retrieved from communication backend");

    }

    double runtime::communicate_test(void* pointer, int64_t size)
    {
        if (!is_disagg_execution_mode(runtime_mode, comm_profile_enabled))
        {
            Logger::get_instance().log(LOG_INFO, "Communication test skipped for non-disaggregation mode");
            return 0.0;
        }
        if (pointer == nullptr || size <= 0) {
            return 0.0;
        }
        return comm_handler->communicate_test(pointer, size, rank, device_id);
    }

    bool runtime::barrier()
    {
        if (!is_disagg_execution_mode(runtime_mode, comm_profile_enabled))
        {
            Logger::get_instance().log(LOG_INFO, "Barrier skipped for non-disaggregation mode");
            return true;
        }
        return comm_handler->barrier(rank);
    }

    void runtime::synchronize()
    {
        cudaStreamSynchronize(getStream()); // Synchronize the default stream
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Logger::get_instance().log(LOG_ERROR, "CUDA error during device synchronization: %s", cudaGetErrorString(err));
            throw std::runtime_error("CUDA error during device synchronization");
        }
    }

    cudaStream_t& runtime::getStream()
    {
        std::lock_guard<std::mutex> lock(stream_mutex);
        int current_device = -1;
        cudaGetDevice(&current_device);
        if (current_device != device_id) {
            cudaSetDevice(device_id);
        }
        std::thread::id this_id = std::this_thread::get_id();
        if (scheduler.asyncStream_Map.find(this_id) == scheduler.asyncStream_Map.end()) {
            int stream_index = allocStream() % scheduler.stream_list.size();
            scheduler.asyncStream_Map[this_id] = scheduler.stream_list[stream_index];
            Logger::get_instance().log(LOG_INFO, "Allocated new stream for thread %zu", this_id); 
        }
        return scheduler.asyncStream_Map[this_id];
    }

    void runtime::setWorkerId(int thread_id)
    {
        if (!is_disagg_execution_mode(runtime_mode, comm_profile_enabled)) {
            return;
        }
        comm_handler->setWorkerId(thread_id);
        Logger::get_instance().log(LOG_INFO, "Communication handler ID set to %d for thread %zu", thread_id, std::this_thread::get_id());
    }
}
