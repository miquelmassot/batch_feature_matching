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
#include <iostream>

int main(int argc, const char **argv) {
  // Check the value of argc.
  // If not enough parameters have been passed, inform user and exit.
  std::string path, format;
  if (argc < 5) {
    // Inform the user of how to use the program
    std::cout << "Usage is -d <ImageDir> -f <ImageFormat>\nExample: "
      << argv[0] << " -d /tmp -f jpg\n";
    return -1;
  } else {
    for (int i = 1; i < argc - 1; i++) {
      if (argv[i] == "-d") {
        // We know the next argument *should* be the filename:
        path = std::string(argv[i + 1]);
      } else if (argv[i] == "-f") {
        format = std::string(argv[i + 1]);
      } else {
        std::cout << "Not enough or invalid arguments, please try again.\n";
        return -1;
      }
    }
  }

  BatchFeatureMatcher bfm(path, format);

  return 0;
}