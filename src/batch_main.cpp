// The MIT License (MIT)

// Copyright (c) 2015 Miquel Massot Campos

//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.

#include <batch_feature_matching/batch_feature_matcher.h>
#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

int main(int argc, const char **argv) {
  // Check the value of argc.
  // If not enough parameters have been passed, inform user and exit.
  std::string path, format;

  po::options_description description("MyTool Usage");

  description.add_options()
    ("help,h", "Display this help message")
    ("path,p", po::value<std::string>(), "Image folder location")
    ("format,f", po::value<std::string>(), "Image format (bmp, jpg, png...)");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(description).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << description;
  }

  if (vm.count("path") && vm.count("format")) {
    path = vm["path"].as<std::string>();
    format = vm["format"].as<std::string>();
    std::cout << "Setting path:   " << path << "\n"
              << "Setting format: " << format << std::endl;
    BatchFeatureMatcher bfm(path, format);
  } else {
    std::cout << description;
  }

  return 0;
}