#pragma once
#include <TokensDataset.hpp>
#include <Loader.hpp>
#include <Augmenter.hpp>
#include <Config.hpp>

class DataManager
{
public:
    DataManager(const Config& config) : dModel(config.dModel)
    {
        loadData(config.dataPath, config.augmentPath);
    }

    auto createDataLoaders(size_t batchSize) const
    {
        auto dataset = TokensDataset(augmentedData, tokenizer, dModel, true)
            .map(torch::data::transforms::Stack<>());
        
        auto testDataset = TokensDataset(augmentedData, tokenizer, dModel, true)
            .map(torch::data::transforms::Stack<>());

        auto loader = torch::data::make_data_loader(
            std::move(dataset),
            torch::data::DataLoaderOptions()
                .batch_size(batchSize)
                .workers(12)
        );

        auto testLoader = torch::data::make_data_loader(
            std::move(testDataset),
            torch::data::DataLoaderOptions()
                .batch_size(1)
        );

        return std::pair{ std::move(loader), std::move(testLoader) };
    }

    const Tokenizer& getTokenizer() const
    {
        return tokenizer;
    }

    const std::vector<std::string>& getAugmentedData() const
    {
        return augmentedData;
    }

private:
    void loadData(const std::string& dataPath, const std::string& augmentPath)
    {
        Loader dialogueLoader(dataPath);
        Loader augmentLoader(augmentPath);

        const auto& data = dialogueLoader.getData();
        const auto& augment = augmentLoader.getData();

        Augmenter augmenter(data, augment);
        augmentedData = augmenter.getAugmented();

        tokenizer.tokenize(data);
        tokenizer.tokenize(augment);
    }

private:
    Tokenizer tokenizer;
    std::vector<std::string> augmentedData;
    size_t dModel;
};
