/* Copyright © 2023 MetaX Integrated Circuits (Shanghai) Co.,Ltd. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 *  1. Redistributions of source code must retain the above copyright notice, this list of
 *     conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice, this list of
 *     conditions and the following disclaimer in the documentation and/or other materials
 *     provided with the distribution.
 *  3. Neither the name of the copyright holder nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without specific prior written
 *     permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef COMMON_MCRTC_HELPER_H_

#define COMMON_MCRTC_HELPER_H_ 1

#include <mc_runtime.h>
#include <helper_maca.h>
#include <helper_string.h>
#include <mcrtc.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define MCRTC_SAFE_CALL(Name, x)                                      \
    do                                                                \
    {                                                                 \
        mcrtcResult result = x;                                       \
        if (result != MCRTC_SUCCESS)                                  \
        {                                                             \
            std::cerr << "\nerror: " << Name << " failed with error " \
                      << mcrtcGetErrorString(result);                 \
            exit(1);                                                  \
        }                                                             \
    } while (0)

void compileFileToBitcode(char *filename, int argc, char **argv, char **bitcodeResult,
                        size_t *bitcodeResultSize, int requiresCGheaders)
{
    std::ifstream inputFile(filename,
                            std::ios::in | std::ios::binary | std::ios::ate);

    if (!inputFile.is_open())
    {
        std::cerr << "\nerror: unable to open " << filename << " for reading!\n";
        exit(1);
    }

    std::streampos pos = inputFile.tellg();
    size_t inputSize = (size_t)pos;
    char *memBlock = new char[inputSize + 1];

    inputFile.seekg(0, std::ios::beg);
    inputFile.read(memBlock, inputSize);
    inputFile.close();
    memBlock[inputSize] = '\x0';

    int numCompileOptions = 0;

    char *compileParams[1];

    if (requiresCGheaders)
    {
        std::string compileOptions;
        char HeaderNames[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(HeaderNames, sizeof(HeaderNames), "%s", "maca_cooperative_groups.h");
#else
        snprintf(HeaderNames, sizeof(HeaderNames), "%s", "maca_cooperative_groups.h");
#endif

        compileOptions = "--include-path=";

        std::string path = sdkFindFilePath(HeaderNames, argv[0]);
        if (!path.empty())
        {
            std::size_t found = path.find(HeaderNames);
            path.erase(found);
        }
        else
        {
            printf(
                "\nmaca_cooperative_groups headers not found, please install it in %s "
                "sample directory..\n Exiting..\n",
                argv[0]);
        }
        compileOptions += path.c_str();
        compileParams[numCompileOptions] = reinterpret_cast<char *>(
            malloc(sizeof(char) * (compileOptions.length() + 1)));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 1),
                  "%s", compileOptions.c_str());
#else
        snprintf(compileParams[numCompileOptions], compileOptions.size(), "%s",
                 compileOptions.c_str());
#endif
        numCompileOptions++;
    }

    // compile
    mcrtcProgram prog;
    MCRTC_SAFE_CALL("mcrtcCreateProgram",
                    mcrtcCreateProgram(&prog, memBlock, filename, 0, NULL, NULL));

    mcrtcResult res = mcrtcCompileProgram(prog, numCompileOptions, compileParams);

    // dump log
    size_t logSize;
    MCRTC_SAFE_CALL("mcrtcGetProgramLogSize",
                    mcrtcGetProgramLogSize(prog, &logSize));
    char *log = reinterpret_cast<char *>(malloc(sizeof(char) * logSize + 1));
    MCRTC_SAFE_CALL("mcrtcGetProgramLog", mcrtcGetProgramLog(prog, log));
    log[logSize] = '\x0';

    if (strlen(log) >= 2)
    {
        std::cerr << "\n compilation log ---\n";
        std::cerr << log;
        std::cerr << "\n end log ---\n";
    }

    free(log);

    MCRTC_SAFE_CALL("mcrtcCompileProgram", res);

    size_t bitcodeSize;
    MCRTC_SAFE_CALL("mcrtcGetBitcodeSize", mcrtcGetBitcodeSize(prog, &bitcodeSize));
    char *bitcode = new char[bitcodeSize];
    MCRTC_SAFE_CALL("mcrtcGetBitcode", mcrtcGetBitcode(prog, bitcode));
    *bitcodeResult = bitcode;
    *bitcodeResultSize = bitcodeSize;

    for (int i = 0; i < numCompileOptions; i++)
    {
        free(compileParams[i]);
    }
}

mcModule_t loadCode(char *code, int argc, char **argv)
{
    mcModule_t module;
    checkMacaErrors(mcModuleLoadData(&module, code));
    free(code);
    return module;
}

#endif // COMMON_MCRTC_HELPER_H_
