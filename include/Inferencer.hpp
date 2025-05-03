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
        if(this->userInput == "exit")
            return "exit";

        if(this->userInput == "reset")
        {
            this->context.clear();
            std::println("Context cleared.\n");

            return "reset";
        }
        
        auto pos = this->userInput.find("temperature");
        if(pos != std::string::npos)
        {
            this->temperature = std::stof(this->userInput.substr(pos + 12));
            std::println("Temperature set to {}.\n", this->temperature);

            return "temperature";
        }

        return "";
    }

    void processInput()
    {
        if(this->userInput != "pass")
        {
            this->context += " [USER] " + this->userInput;
            this->context += " [ASSISTANT] ";
        }
    }

    void generateResponse()
    {
        auto tokens = this->tokenizer.encode(this->context);
        
        if(tokens.size() > this->maxSize)
            tokens.erase(tokens.begin(), tokens.begin() + tokens.size() - this->maxSize);

        auto output = this->transformer->generate(std::move(tokens), this->maxSize, 128, this->temperature);
        
        std::print("\nPredicted: ");
        for(const auto& i : output)
        {
            if(i != 0 && i != 2)
            {
                auto token = this->tokenizer.decode(i);
                std::print("{} ", token);
                this->context += token + ' ';
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
