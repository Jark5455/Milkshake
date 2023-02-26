#include "stockframe.h"

#include <cstring>
#include <iostream>
#include <unistd.h>
#include <utility>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sstream>
#include <random>

#define BUF_SIZE 4096
#define KEY getenv("ALPACA_KEY")
#define SECRET getenv("ALPACA_SECRET")

StockFrame::StockFrame(std::vector<std::string> tickers) : tickers{std::move(tickers)} {
    establishTLSConnection("data.alpaca.markets");
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

void StockFrame::establishTLSConnection(const std::string &host) {

    std::cout << "Establishing TCP Connection" << '\n';

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

    std::random_device engine;
    std::vector<uint8_t> clientHello =
    {
        //Record Header
        0x16, //type: handshake
        0x03, 0x01, //TLS version:
        0x00, 0x00, //HandShake Length e.g clientHello.length - 5

        //Handshake Header
        0x01, //type: client hello
        0x00, 0x00, 0x00, //handshake data len e.g clientHello.length - 9

        // Client Version
        0x03, 0x03, // TLS 1.2 //actual TLS version client uses

        // Client Random
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),
        static_cast<uint8_t>(engine() % 255),

        //Session ID
        // This ID is used to logon back from a previous session.
        // This eliminates certain computations as per what I read.
        // Here we use zero to say this is a fresh session with the server.
        0x00,

        // Cypher Suites
        // A list of Cypher Suite used to do the encryption.
        // Cypher Suites are represented by two bytes.
        // Example 0xCCA8 is TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256 Cypher Suite.

        0x00, 0x02, // count of bytes

        // TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
        0xc0, 0x2b,

        // Compression Methods
        // A list of compression methods.
        // Here we have only one compression method which is zero for no compression.
        0x01, //length of compression methods
        0x00, // methods used
    };

    std::vector<uint8_t>  extensions{};

    std::vector<uint8_t> serverNameExtension{};
    // type: server name
    serverNameExtension.push_back(0x00);
    serverNameExtension.push_back(0x00);

    // extension length
    serverNameExtension.push_back(((host.length() + 5) >> 8) & 0xff);
    serverNameExtension.push_back(((host.length() + 5) >> 0) & 0xff);

    // server name list length
    serverNameExtension.push_back(((host.length() + 3) >> 8) & 0xff);
    serverNameExtension.push_back(((host.length() + 3) >> 0) & 0xff);

    // type hostname
    serverNameExtension.push_back(0x00);

    // name length
    serverNameExtension.push_back(((host.length()) >> 8) & 0xff);
    serverNameExtension.push_back(((host.length()) >> 0) & 0xff);

    for (char character : host) {
        serverNameExtension.push_back(character);
    }

    extensions.insert(extensions.end(), serverNameExtension.begin(), serverNameExtension.end());

    std::vector<uint8_t> elliptical_point_formats = {
            // type
            0x00, 0x0b,

            // lengths
            0x00, 0x02,
            0x01,

            // uncompressed format
            0x00
    };

    extensions.insert(extensions.end(), elliptical_point_formats.begin(), elliptical_point_formats.end());

    std::vector<uint8_t> supported_groups = {

            // type
            0x00, 0x0a,

            // lengths
            0x00, 0x04,
            0x00, 0x02,

            // secp256r1
            0x00, 0x17
    };

    extensions.insert(extensions.end(), supported_groups.begin(), supported_groups.end());


    std::vector<uint8_t> application_layer_protocol_negotiation_extension{};

    // type : app layer protocol negotiation
    application_layer_protocol_negotiation_extension.push_back(0x00);
    application_layer_protocol_negotiation_extension.push_back(0x10);


    // length
    application_layer_protocol_negotiation_extension.push_back(0x00);
    application_layer_protocol_negotiation_extension.push_back(0x0b);

    // ALPN data length
    application_layer_protocol_negotiation_extension.push_back(0x00);
    application_layer_protocol_negotiation_extension.push_back(0x09);

    std::string ALPN = "http/1.0";

    // string length
    application_layer_protocol_negotiation_extension.push_back(ALPN.length());

    for (char character : ALPN) {
        application_layer_protocol_negotiation_extension.push_back(character);
    }

    extensions.insert(extensions.end(), application_layer_protocol_negotiation_extension.begin(), application_layer_protocol_negotiation_extension.end());

    std::vector<uint8_t> signature_algorithms_extension = {

            // type
            0x00, 0x0d,

            // lengths
            0x00, 0x04,
            0x00, 0x02,

            // ecdsa_secp256r1_sha256
            0x04, 0x03
    };

    extensions.insert(extensions.end(), signature_algorithms_extension.begin(), signature_algorithms_extension.end());

    // extensions length
    clientHello.push_back((extensions.size() >> 8) & 0xff);
    clientHello.push_back((extensions.size() >> 0) & 0xff);

    clientHello.insert(clientHello.end(), extensions.begin(), extensions.end());

    // lengths
    clientHello[3] = ((clientHello.size() - 5) >> 8) & 0xff;
    clientHello[4] = ((clientHello.size() - 5) >> 0) & 0xff;

    // idk what this bit does
    clientHello[6] = 0;

    // lengths
    clientHello[7] = ((clientHello.size() - 9) >> 8) & 0xff;
    clientHello[8] = ((clientHello.size() - 9) >> 0) & 0xff;

    size_t total = clientHello.size();
    size_t sent = 0;

    while (sent < total) {
        ssize_t bytes = write(sockfd, clientHello.data() + sent, total - sent);
        if (bytes < 0) throw std::runtime_error("error writing message to socket");
        sent += bytes;

        std::cout << "Sent " << sent << " out of " <<  total << " bytes \n";
    }

    uint8_t metadata[5];
    memset(metadata, 0, 5);

    total = 5;
    ssize_t received = 0;

    while (received < total) {
        ssize_t bytes = read(sockfd, metadata + received, total - received);

        if (bytes < 0) {
            throw std::runtime_error("Failed to read socket");
        }

        received += bytes;
        std::cout << "Read " << received << " out of " << total << " bytes \n";
    }

    // content type: handshake
    assert(metadata[0] == 22);

    // resize based on length bytes
    ssize_t length = metadata[3] << 8 | metadata[4];
    uint8_t serverHelloData[length];

    received = 0;

    while (received < length) {
        ssize_t bytes = read(sockfd, serverHelloData + received, length - received);

        if (bytes < 0) {
            throw std::runtime_error("Failed to read socket");
        }

        received += bytes;
        std::cout << "Read " << received << " out of " << length << " bytes \n";
    }

    std::vector<uint8_t> serverHello{};
    serverHello.insert(serverHello.end(), metadata, metadata + 5);
    serverHello.insert(serverHello.end(), serverHelloData, serverHelloData + length);

    memset(metadata, 0, 5);
    total = 5;
    received = 0;

    while (received < total) {
        ssize_t bytes = read(sockfd, metadata + received, total - received);

        if (bytes < 0) {
            throw std::runtime_error("Failed to read socket");
        }

        received += bytes;
        std::cout << "Read " << received << " out of " << total << " bytes \n";
    }

    // content type: handshake
    assert(metadata[0] == 22);

    // resize based on length bytes
    length = metadata[3] << 8 | metadata[4];
    uint8_t serverCertData[length];

    received = 0;

    while (received < length) {
        ssize_t bytes = read(sockfd, serverCertData + received, length - received);

        if (bytes < 0) {
            throw std::runtime_error("Failed to read socket");
        }

        received += bytes;
        std::cout << "Read " << received << " out of " << length << " bytes \n";
    }

    std::vector<uint8_t> serverCert{};
    serverCert.insert(serverCert.end(), metadata, metadata + 5);
    serverCert.insert(serverCert.end(), serverCertData, serverCertData + length);



    close(sockfd);
}
