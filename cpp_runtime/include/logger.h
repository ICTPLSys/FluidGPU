// log.h
#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdarg>
#include <filesystem>

#define DEBUG 1
#define NORMAL 0

enum LogLevel {
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR,
    
    
};

class Logger {
private:
    std::ofstream log_file_;
    int rank_id = 0; // Default rank ID, can be set later
    int mode = DEBUG; // Default mode is NORMAL
    std::string get_time_string() {
        std::time_t now = std::time(nullptr);
        char buf[80];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        return std::string(buf);
    }

    std::string level_to_string(LogLevel level) {
        switch (level) {
            case LOG_INFO: return "INFO";
            case LOG_WARN: return "WARN";
            case LOG_ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }

public:
    Logger() {
        const std::string log_dir = "./log";
        if (!std::filesystem::exists(log_dir)) {
            std::filesystem::create_directory(log_dir);
        }
        const std::string log_file_path = log_dir + "/log_" + get_time_string() + ".txt";
        log_file_.open(log_file_path, std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "Failed to open log file: " << log_file_path << std::endl;
        }
    }
    
    ~Logger()
    {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }

    static Logger& get_instance() {
        static Logger instance;
        return instance;
    }

    void set_rank(int rank) {
        rank_id = rank;
    }

    void setLogMode(int m) {
        mode = m;
    }
    
    void log(LogLevel level, const char* format, ...)
    {
        if (mode == NORMAL)
            return; 
        if (log_file_.is_open()) {
            char buffer[1024];
            va_list args;
            va_start(args, format);
            vsnprintf(buffer, sizeof(buffer), format, args);
            va_end(args);

            std::string time_str = get_time_string();
            std::string level_str = level_to_string(level);
            log_file_ << "[" << time_str << "] [" << level_str << "] [Rank " << rank_id << "] " << buffer << std::endl;
            log_file_.flush(); // 确保日志立即写入文件
        }
    }
};

// 宏定义，方便使用
#define LOG_INFO(format, ...) logger_.log(LOG_INFO, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) logger_.log(LOG_WARN, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) logger_.log(LOG_ERROR, format, ##__VA_ARGS__)


#endif // LOGGER_H