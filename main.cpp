#include <TokensDataset.hpp>
#include <Transformer.hpp>
#include <Trainer.hpp>
#include <Tester.hpp>

#include <fstream>

int main()
{
    using std::operator""s;

    const size_t batchSize = 32;
    const size_t epochs = 500;
    const size_t maxSize = 64;
    const size_t maxSeq = 64;
    const float learningRate = 0.001;
    const float temperature = 0.5;
    const bool load = true;

    std::vector<std::string> data;

    std::ifstream file("../data/data.txt");
    std::string line;

    while(std::getline(file, line))
        data.push_back(line);

    Tokenizer tokenizer;
    tokenizer.tokenize(data);

    std::srand(std::time(0));
    torch::manual_seed(std::rand());

    auto dataset = TokensDataset(data, tokenizer, maxSize, true)
        .map(torch::data::transforms::Stack<>());
    
    auto testDataset = TokensDataset(data, tokenizer, maxSize, true)
        .map(torch::data::transforms::Stack<>());

    auto loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(batchSize)
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

        context += " [USER] " + userInput;

        auto tokens = tokenizer.encode(context);
        
        if(tokens.size() > maxSize)
            tokens.erase(tokens.begin(), tokens.begin() + tokens.size() - maxSize);

        tokens.resize(maxSize, 0);

        context += " [ASSISTANT] ";

        std::cout << "Encoded: ";
        for(const auto& i : tokens)
            if(i != 0)
                std::cout << tokenizer.decode(i) << ' ';

        std::cout << "\n\n";

        auto output = transformer->generate(tokens, maxSize, 128, temperature);
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
