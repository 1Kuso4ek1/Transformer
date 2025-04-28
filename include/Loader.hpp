#pragma once
#include <vector>
#include <string>
#include <fstream>

class Loader
{
public:
    Loader(const std::string& filename) : file(filename)
    {
        load();
    }

    void load()
    {
        std::string line;

        if(!file.is_open())
            return;
        
        while(std::getline(file, line))
            data.push_back(line);
    }

    const std::vector<std::string>& getData() const
    {
        return data;
    }

private:
    std::vector<std::string> data;
    std::ifstream file;
};
