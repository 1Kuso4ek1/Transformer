#include <TokensDataset.hpp>
#include <Transformer.hpp>
#include <Trainer.hpp>
#include <Tester.hpp>
#include <Loader.hpp>
#include <Augmenter.hpp>
#include <Config.hpp>
#include <CommandLine.hpp>
#include <Inferencer.hpp>
#include <DataManager.hpp>

bool trainingConfirmation(const Config& config)
{
    std::println("Training with the following configuration:");
    std::println("Batch size: {}", config.trainBatch);
    std::println("Epochs: {}", config.epochs);
    std::println("Model size: {}", config.dModel);
    std::println("Max sequence length: {}", config.maxSeq);
    std::println("Learning rate: {}", config.learningRate);

    char choice;
    std::print("\nContinue? (y/n): ");
    std::cin >> choice;

    return choice == 'y' || choice == 'Y';
}

void training(
    const DataManager& dataManager,
    std::shared_ptr<Transformer> transformer,
    const Config& config
)
{
    if(!trainingConfirmation(config))
    {
        std::println("Training aborted.\n");
        return;
    }

    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.clear();

    auto [trainLoader, testLoader] = dataManager.createDataLoaders(config.trainBatch);
    
    Trainer(std::move(trainLoader), transformer, config).train();
    Tester(std::move(testLoader), transformer).test(dataManager.getTokenizer());
}

int main(int argc, char** argv)
{
    CommandLine commandLine(argc, argv);

    Config config { 32, 30, 128, 128, 0.002f, false };
    if(!config.loadFromFile(commandLine.getConfigPath()))
        std::println("Failed to load config file. Using default values.");

    std::srand(std::time(0));
    torch::manual_seed(std::rand());

    DataManager dataManager(config);
    
    auto transformer = std::make_shared<Transformer>(dataManager.getTokenizer().size(), config.dModel, config.maxSeq);
    if(config.load)
        torch::load(transformer, config.modelPath);

    if(commandLine.isTrainingMode())
        training(dataManager, transformer, config);
    
    Inferencer(dataManager.getTokenizer(), transformer, config.dModel).run();

    if(!config.saveToFile(commandLine.getConfigPath()))
        std::println("Failed to save config file.");
}
