#include "text_generator.hpp"

#include <iostream>
#include <string>

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end = s.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

int main(int argc, char* argv[]) {
    TextGenerator gen;

    std::string dataDir = "data";
    std::string embeddingsFile;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a.rfind(".txt") != std::string::npos && a.find("glove") != std::string::npos)
            embeddingsFile = a;
        else if (a.find(".txt") == std::string::npos && a.find("glove") == std::string::npos)
            dataDir = a;
    }
    gen.loadDataset(dataDir, embeddingsFile);
    gen.loadBlueprint("blueprint");
    gen.loadNames("data/names.txt");
    gen.loadTemplates("data/templates");

    if (gen.empty()) {
        std::cerr << "No data loaded.\n";
        return 1;
    }

    std::cout << "\n=== Text Generator ===\n";

    while (true) {
        auto genres = gen.getGenres();
        std::cout << "\nGenres: ";
        for (const auto& g : genres) std::cout << g << " ";
        std::cout << "\nGenre (or 'quit'): ";

        std::string genre;
        std::getline(std::cin, genre);
        genre = trim(genre);
        if (genre == "quit" || genre == "q") break;

        bool found = false;
        for (const auto& g : genres)
            if (g == genre) { found = true; break; }
        if (!found) {
            std::cout << "Unknown genre.\n";
            continue;
        }

        std::cout << "Word count: ";
        std::string countStr;
        std::getline(std::cin, countStr);

        try {
            int count = std::stoi(countStr);
            std::string text = gen.generate(genre, count);
            std::cout << "\n--- Generated ---\n" << text << "\n";

            auto scores = gen.classify(text);
            if (!scores.empty()) {
                std::cout << "\n--- Best genre ---\n  " << scores.front().first
                          << ": " << scores.front().second << "\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << "\n";
        }
    }

    std::cout << "Done.\n";
    return 0;
}
