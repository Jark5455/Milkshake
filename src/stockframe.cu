#include "stockframe.h"

#include <cstring>
#include <iostream>
#include <unistd.h>
#include <utility>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sstream>

#define BUF_SIZE 4096
#define KEY getenv("ALPACA_KEY")
#define SECRET getenv("ALPACA_SECRET")

StockFrame::StockFrame(std::vector<std::string> tickers) : tickers{std::move(tickers)} {
    updateStockFrame();
}

StockFrame::~StockFrame() {
}

void StockFrame::updateStockFrame() {
    std::ostringstream queryString;
    queryString << "GET /v2/stocks/bars?";

    // make changeable later
    queryString << "timeframe=1Min";

    // add other params here

    queryString << "&symbols=";
    for (const std::string& ticker : tickers) {
        queryString << ticker << ",";
    }

    // overwrite last comma
    queryString.seekp(-1, std::ios_base::end);

    queryString << " HTTP/1.0\r\n";
    queryString << "APCA-API-KEY-ID: " << KEY << "\r\n";
    queryString << "APCA-API-SECRET-KEY: " << SECRET << "\r\n";
    queryString << "\r\n";

    std::cout << send_request("data.alpaca.markets", queryString.str());
}

void StockFrame::addTicker(const std::string& str) {
    tickers.push_back(str);
}

std::string StockFrame::send_request(const std::string& host, const std::string& request) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) throw std::runtime_error("error opening socket");

    struct timeval tv{5};
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

    auto server = gethostbyname(host.c_str());
    if (server == nullptr) throw std::runtime_error("error no such host");

    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(443);
    memcpy(&serv_addr.sin_addr.s_addr, server->h_addr, server->h_length);

    if (connect(sockfd, reinterpret_cast<const sockaddr *>(&serv_addr), sizeof(serv_addr)) < 0) {
        throw std::runtime_error("error connecting to socket");
    }

    size_t total = request.length();
    size_t sent = 0;

    while (sent < total) {
        ssize_t bytes = write(sockfd, request.c_str() + sent, total - sent);
        if (bytes < 0) throw std::runtime_error("error writing message to socket");
        sent += bytes;

        std::cout << "Sent " << sent << " out of " <<  total << " bytes \n";
    }

    std::string response;
    ssize_t received = 0;

    while (true) {
        char buf[BUF_SIZE];
        ssize_t bytes = recv(sockfd, buf, BUF_SIZE, 0);

        if (bytes < 0) throw std::runtime_error("error reading response from socket");
        if (bytes == 0) break;

        received += bytes;
        response.append(buf, bytes);
    }

    close(sockfd);
    return {response.begin(), response.end()};
}