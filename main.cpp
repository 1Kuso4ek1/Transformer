#include <TokensDataset.hpp>
#include <Transformer.hpp>
#include <Trainer.hpp>
#include <Tester.hpp>
#include <Loader.hpp>
#include <Augmenter.hpp>

struct Config
{
    int batchSize, epochs;
    size_t maxSize, maxSeq;
    float learningRate;
    bool load;
};

void inference(
    const Tokenizer& tokenizer,
    std::shared_ptr<Transformer> transformer,
    size_t maxSize
)
{
    float temperature = 0.5;

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

        auto output = transformer->generate(std::move(tokens), maxSize, 128, temperature);
        std::cout << "\nPredicted: ";
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

int main()
{
    const Config config { 32, 10, 64, 100, 0.002, true };

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

    auto dataset = TokensDataset(augmentedData, tokenizer, config.maxSize, true)
        .map(torch::data::transforms::Stack<>());
    
    auto testDataset = TokensDataset(augmentedData, tokenizer, config.maxSize, true)
        .map(torch::data::transforms::Stack<>());

    auto loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(config.batchSize)
            .workers(12)
    );

    auto testLoader = torch::data::make_data_loader(
        std::move(testDataset),
        torch::data::DataLoaderOptions()
            .batch_size(1)
    );

    auto transformer = std::make_shared<Transformer>(tokenizer.size(), config.maxSize, config.maxSeq);

    if(config.load)
        torch::load(transformer, "model.pt");

    Trainer(std::move(loader), transformer, { config.epochs, config.batchSize, config.learningRate, config.load }).train();
    Tester(std::move(testLoader), transformer, { 1 }).test(tokenizer);

    inference(tokenizer, transformer, config.maxSize);
}
