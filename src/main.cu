#include <iostream>
#include <cstdlib>
#include <cudnn.h>

#include "buffer.h"
#include "stockframe.h"

int main() {
    StockFrame stockFrame = StockFrame({"AAPL"});
    return 0;
}
