// This exists because cuda is unable to compile boost headers, I am precompiling this header in gcc and using it in cuda

#pragma once

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/bind.hpp>