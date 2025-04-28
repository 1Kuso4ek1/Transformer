#pragma once
#include <c10/core/Device.h>

namespace global
{

static at::Device device(c10::kCPU);
    
}

inline std::string separatePunctuation(const std::string& str)
{
    std::string modified;

    for(int j = 0; j < str.size(); j++)
    {
        if(std::ispunct(str[j]) && !modified.empty() && str[j] != '[' && str[j] != ']')
        {
            modified += (modified.back() != ' ' ? std::string(1, ' ') : "") + str[j];

            if(j < str.size() - 1 && str[j + 1] != ' ')
                modified += ' ';
        }
        else
            modified += str[j];
    }

    return modified;
}
