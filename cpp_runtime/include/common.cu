#include "common.h"
#include <cerrno>
#include <cstring>

int socket_connect(const std::string& ip_port, int rank) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return -1;
    }

    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(std::stoi(ip_port.substr(ip_port.find(":") + 1)));
    inet_pton(AF_INET, ip_port.substr(0, ip_port.find(":")).c_str(), &addr.sin_addr);

    if (rank == 0) {
        // Server setup
        if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            throw std::runtime_error("Failed to bind socket");
        }
        if (listen(sockfd, 1) < 0) {
            throw std::runtime_error("Failed to listen on socket");
        }
        int client_sock = accept(sockfd, nullptr, nullptr);
        if (client_sock < 0) {
            throw std::runtime_error("Failed to accept connection");
        }
        close(sockfd);
        sockfd = client_sock;
    } 
    else {
        // Client setup
        while (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "Failed to connect to server " << ip_port
                      << " as rank " << rank
                      << ", errno=" << std::strerror(errno)
                      << ", retrying..." << std::endl;
            sleep(1);  // Wait before retrying
        }
    }

    return sockfd;
}
