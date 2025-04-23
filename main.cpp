#include <TokensDataset.hpp>
#include <Transformer.hpp>
#include <Trainer.hpp>
#include <Tester.hpp>

#include <fstream>

int main()
{
    using std::operator""s;

    const size_t batchSize = 128;
    const size_t epochs = 300;
    const size_t maxSize = 32;
    const size_t maxSeq = 100;
    const float learningRate = 0.003;
    const float temperature = 0.7;

    std::vector data =
    {
        "привет, как дела у тебя, дружище? все хорошо!"s,
        "трансформеры это круто, так ведь? да, ты прав."s,
        "внимание точно работает? да, именно так."s,
        "мяу мур мурмяу мяу мяу мяу мяу мур"s,
        "я - прикольная нейросеть, правда? ну да, реально."s,
        "один, два, три, четыре, пять, шесть, семь, восемь"s
    };

    std::ifstream file("../data/data.txt");
    std::string line;

    while(std::getline(file, line))
        data.push_back(line);

    Tokenizer tokenizer;
    tokenizer.tokenize(data);

    std::srand(std::time(0));
    torch::manual_seed(std::rand());

    auto dataset = TokensDataset(data, tokenizer, maxSize)
        .map(torch::data::transforms::Stack<>());
    
    auto testDataset = TokensDataset(data, tokenizer, maxSize)
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
    //torch::load(transformer, "model.pt");

    Trainer(std::move(loader), transformer, { epochs, batchSize, learningRate/* , true */ }).train();
    Tester(std::move(testLoader), transformer, { 1 }).test(tokenizer);

    std::string userInput;

    while(userInput != "exit")
    {
        std::cout << "> ";
        std::getline(std::cin, userInput);

        auto tokens = tokenizer.encode(userInput);
        std::cout << "Encoded: ";
        for(const auto& i : tokens)
            if(i != 0)
                std::cout << tokenizer.decode(i) << ' ';

        std::cout << "\n\n";

        auto output = transformer->generate(std::move(tokens), maxSize, 8, temperature);
        std::cout << "Predicted: ";
        for(const auto& i : output)
            if(i != 0)
                std::cout << tokenizer.decode(i) << ' ';

        std::cout << "\n\n";
    }
}
