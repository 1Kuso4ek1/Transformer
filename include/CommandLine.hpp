#pragma once
#include <argparse/argparse.hpp>

class CommandLine
{
public:
    CommandLine(int argc, char** argv)
    {
        program.add_argument("--config")
            .help("path to config file")
            .default_value("config.json");
        
        program.add_argument("--mode")
            .help("train or inference mode")
            .default_value("train")
            .choices("train", "inference");
        
        program.parse_args(argc, argv);
    }

    std::string getConfigPath() const {
        return program.get<std::string>("--config");
    }

    bool isTrainingMode() const {
        return program.get<std::string>("--mode") == "train";
    }

private:
    argparse::ArgumentParser program{"transformer"};
};
