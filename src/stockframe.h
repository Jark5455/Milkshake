//
// Created by yashr on 2/21/23.
//

#include <arrow/table.h>

// boost libs in disguise
#include "pch.h"

#pragma once

class StockFrame {
    public:
        StockFrame(std::vector<std::string> tickers);
        ~StockFrame();

        // async download stock data from alpaca
        void asyncDownloadAggregateData();
        void waitAggregateData();
        void clearAggregateData();

        // actually calculate all indicators and update the stockframe
        void updateStockFrame();

        void addTicker(const std::string& str);
    private:
        void asyncResolveRequest(const boost::system::error_code& ec, boost::asio::ip::tcp::resolver::results_type results);
        void asyncDownloadPage(std::string url);
        boost::asio::io_context context;

        std::unique_ptr<arrow::Table> stockFrame;
        std::unique_ptr<arrow::Table> aggregateData;
        std::vector<std::string> tickers;
};