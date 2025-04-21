#include <TokensDataset.hpp>
#include <Transformer.hpp>
#include <Trainer.hpp>
#include <Tester.hpp>

#include <fstream>

int main()
{
    using std::operator""s;

    const size_t maxSize = 32;

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

    auto dataset = TokensDataset(data, tokenizer, maxSize)
        .map(torch::data::transforms::Stack<>());
    
    auto testDataset = TokensDataset(data, tokenizer, maxSize)
        .map(torch::data::transforms::Stack<>());

    auto loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(128)
    );

    auto testLoader = torch::data::make_data_loader(
        std::move(testDataset),
        torch::data::DataLoaderOptions()
            .batch_size(1)
    );

    auto transformer = std::make_shared<Transformer>(tokenizer.size(), maxSize);
    //torch::load(transformer, "model.pt");

    Trainer(std::move(loader), transformer, { 400, 128, 0.003 }).train();
    Tester(std::move(testLoader), transformer, { 1 }).test(tokenizer);

    std::string userInput;

    while(userInput != "exit")
    {
        std::cout << "> ";
        std::getline(std::cin, userInput);

        auto tokens = tokenizer.encode(userInput);
        auto output = transformer->generate(std::move(tokens), maxSize, 3);

        std::cout << "Predicted: ";
        for(const auto& i : output)
            if(i != 0)
                std::cout << tokenizer.decode(i) << ' ';

        std::cout << "\n\n";
    }
}
