#include <TokensDataset.hpp>
#include <Transformer.hpp>
#include <Trainer.hpp>
#include <Tester.hpp>
#include <Loader.hpp>
#include <Augmenter.hpp>

int main()
{
    const size_t batchSize = 64;
    const size_t epochs = 50;
    const size_t maxSize = 64;
    const size_t maxSeq = 100;
    const float learningRate = 0.001;
    const bool load = false;

    float temperature = 0.5;

    Loader dialogueLoader("../data/data.txt");
    Loader augmentLoader("../data/augment.txt");

    const auto& data = dialogueLoader.getData();
    const auto& augment = augmentLoader.getData();

    Augmenter augmenter(data, augment);

    const auto& augmentedData = augmenter.getAugmented();

    Tokenizer tokenizer;
    tokenizer.tokenize(data);
    tokenizer.tokenize(augment);

    std::srand(std::time(0));
    torch::manual_seed(std::rand());

    auto dataset = TokensDataset(augmentedData, tokenizer, maxSize, true)
        .map(torch::data::transforms::Stack<>());
    
    auto testDataset = TokensDataset(augmentedData, tokenizer, maxSize, true)
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

    auto transformer = std::make_shared<Transformer>(tokenizer.size(), maxSize, maxSeq);

    if(load)
        torch::load(transformer, "model.pt");

    Trainer(std::move(loader), transformer, { epochs, batchSize, learningRate, load }).train();
    Tester(std::move(testLoader), transformer, { 1 }).test(tokenizer);

    std::string context, userInput;

    int passIteration{};

    while(userInput != "exit")
    {
        std::cout << "> ";
        std::getline(std::cin, userInput);

        if(userInput == "reset")
        {
            context.clear();
            std::cout << "Context cleared.\n\n";
            continue;
        }
        
        auto pos = userInput.find("temperature");

        if(pos != std::string::npos)
        {
            temperature = std::stof(userInput.substr(pos + 12));
            std::cout << "Temperature set to " << temperature << ".\n\n";
            continue;
        }

        if(userInput != "pass")
            context += " [USER] " + userInput;

        auto tokens = tokenizer.encode(context + " [ASSISTANT] ");
        
        if(tokens.size() > maxSize)
            tokens.erase(tokens.begin(), tokens.begin() + tokens.size() - maxSize);

        if(userInput != "pass")
            context += " [ASSISTANT] ";

        std::cout << "Encoded: ";
        for(const auto& i : tokens)
            if(i != 0)
                std::cout << tokenizer.decode(i) << ' ';

        std::cout << "\n\n";

        auto output = transformer->generate(std::move(tokens), maxSize, 128, temperature);
        std::cout << "Predicted: ";
        for(const auto& i : output)
            if(i != 0 && i != 2)
            {
                auto token = tokenizer.decode(i);
                std::cout << token << ' ';
                context += token + ' ';
            }

        std::cout << "\n\n";
    }
}
