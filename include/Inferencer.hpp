#pragma once

#include <Transformer.hpp>
#include <Tokenizer.hpp>
#include <string>
#include <memory>
#include <print>

class Inferencer {
public:
    Inferencer(const Tokenizer& tokenizer, std::shared_ptr<Transformer> transformer, size_t maxSize)
        : tokenizer(tokenizer), transformer(transformer), maxSize(maxSize), temperature(0.5)
    {}

    void run()
    {
        printInstructions();
        
        while(true)
        {
            std::print("> ");
            std::getline(std::cin, userInput);
            
            auto lastCommand = handleCommands();

            if(lastCommand == "exit")
                break;
            else if(!lastCommand.empty())
                continue;

            processInput();
            generateResponse();
        }
    }

private:
    void printInstructions()
    {
        std::println("Inference mode. Type 'exit' to quit.");
        std::println("Type 'reset' to clear context.");
        std::println("Type 'temperature <value>' to set temperature.");
        std::println("Type 'pass' to skip the input.\n");
    }

    std::string handleCommands()
    {
        if(userInput == "exit")
            return "exit";

        if(userInput == "reset")
        {
            context.clear();
            std::println("Context cleared.\n");

            return "reset";
        }
        
        auto pos = userInput.find("temperature");
        if(pos != std::string::npos)
        {
            temperature = std::stof(userInput.substr(pos + 12));
            std::println("Temperature set to {}.\n", temperature);

            return "temperature";
        }

        return "";
    }

    void processInput()
    {
        if(userInput != "pass")
        {
            context += " [USER] " + userInput;
            context += " [ASSISTANT] ";
        }
    }

    void generateResponse()
    {
        auto tokens = tokenizer.encode(context);
        
        if(tokens.size() > maxSize)
            tokens.erase(tokens.begin(), tokens.begin() + tokens.size() - maxSize);
        
        std::print("\nPredicted: ");
        for(const auto& i : transformer->generate(std::move(tokens), maxSize, 128, temperature))
        {
            if(i != 0 && i != 2)
            {
                auto token = tokenizer.decode(i);
                std::print("{} ", token);
                std::cout.flush();
                
                context += token + ' ';
            }
        }

        std::println("\n");
    }

private:
    const Tokenizer& tokenizer;
    std::shared_ptr<Transformer> transformer;
    
    size_t maxSize;
    float temperature;

    std::string context;
    std::string userInput;
};
