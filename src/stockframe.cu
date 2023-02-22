#include "stockframe.h"

#include <iostream>
#include <utility>

StockFrame::StockFrame(std::vector<std::string> tickers) : tickers{std::move(tickers)} {
    updateStockFrame();
}

StockFrame::~StockFrame() {

}

void StockFrame::asyncDownloadAggregateData() {
    asyncDownloadPage("https://data.alpaca.markets/v2/stocks/bars?symbols=AAPL&timeframe=1Min");
}

void StockFrame::updateStockFrame() {
    asyncDownloadAggregateData();
}

void StockFrame::addTicker(const std::string& str) {
    tickers.push_back(str);
}

void StockFrame::asyncResolveRequest(const boost::system::error_code &ec, boost::asio::ip::tcp::resolver::results_type results) {
    if (!ec) {
        boost::beast::tcp_stream stream(context);
        stream.connect(results);

        boost::beast::http::request<boost::beast::http::string_body> request{boost::beast::http::verb::get, "80", 1.1};
        request.set(boost::beast::http::field::host, results->host_name());
        request.set(boost::beast::http::field::user_agent, BOOST_BEAST_VERSION_STRING);
        boost::beast::http::write(stream, request);

        boost::beast::flat_buffer buffer;
        boost::beast::http::response<boost::beast::http::dynamic_body> response;

        boost::beast::http::read(stream, buffer, response);
        std::cout << response << '\n';

    } else {
        throw std::runtime_error("Failed to resolve endpoint");
    }
}

void StockFrame::asyncDownloadPage(std::string url) {
    boost::asio::ip::tcp::resolver resolver(context);
    resolver.async_resolve(url, "80", boost::bind(&StockFrame::asyncResolveRequest, this, boost::asio::placeholders::error, boost::asio::placeholders::results));
}
