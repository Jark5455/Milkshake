//
// Created by yashr on 2/21/23.
//

#include <arrow/table.h>

#pragma once

class StockFrame {
    public:
        StockFrame(std::vector<std::string> tickers);
        ~StockFrame();

        // actually calculate all indicators and update the stockframe
        void updateStockFrame();

        void addTicker(const std::string& str);
    private:
        void establishTLSConnection(const std::string &host);
        std::string send_request(const std::string &host, const std::string &request);

        std::unique_ptr<arrow::Table> stockFrame;
        std::unique_ptr<arrow::Table> aggregateData;
        std::vector<std::string> tickers;
};