#pragma once
#include <fstream>

#include <nlohmann/json.hpp>

struct Config
{
    int trainBatch = 32, epochs = 30;
    size_t dModel = 128, maxSeq = 128;
    float learningRate = 0.002f;
    bool load = true;

    std::string modelPath = "model.pt";
    std::string optimizerPath = "optimizer.pt";

    std::string dataPath = "../data/data.txt";
    std::string augmentPath = "../data/augment.txt";

    void fromJson(const nlohmann::json& json)
    {
        trainBatch = json.at("trainBatch").get<int>();

        dModel = json.at("dModel").get<size_t>();
        maxSeq = json.at("maxSeq").get<size_t>();
        
        epochs = json.at("epochs").get<int>();
        learningRate = json.at("learningRate").get<float>();
        
        load = json.at("load").get<bool>();

        modelPath = json.at("modelPath").get<std::string>();
        optimizerPath = json.at("optimizerPath").get<std::string>();
        
        dataPath = json.at("dataPath").get<std::string>();
        augmentPath = json.at("augmentPath").get<std::string>();
    }

    nlohmann::json toJson() const
    {
        return {
            { "trainBatch", trainBatch },
            
            { "dModel", dModel },
            { "maxSeq", maxSeq },

            { "epochs", epochs },
            { "learningRate", learningRate },
            
            { "load", load },
            
            { "modelPath", modelPath },
            { "optimizerPath", optimizerPath },
            
            { "dataPath", dataPath },
            { "augmentPath", augmentPath }
        };
    }

    bool saveToFile(const std::string& filename) const
    {
        std::ofstream file(filename);

        if(file.is_open())
        {
            file << toJson().dump(4);
            file.close();

            return true;
        }

        return false;
    }

    bool loadFromFile(const std::string& filename)
    {
        std::ifstream file(filename);

        if(file.is_open())
        {
            nlohmann::json json;
            file >> json;
            file.close();
            fromJson(json);

            return true;
        }

        return false;
    }
};
