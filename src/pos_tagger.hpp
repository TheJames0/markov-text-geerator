#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

class POSTagger {
public:
    POSTagger(const std::string& udpipePath, const std::string& modelPath)
        : udpipePath(udpipePath), modelPath(modelPath) {}

    struct Token {
        std::string word;
        std::string pos;
    };

    std::vector<Token> tag(const std::string& text) const {
        return tagLines({text});
    }

    std::vector<Token> tagLines(const std::vector<std::string>& texts) const {
        if (texts.empty()) return {};

        std::string tmpfile = "/tmp/udpipe_input_XXXXXX";
        int fd = mkstemp(tmpfile.data());
        if (fd == -1) return {};
        FILE* tmp = fdopen(fd, "w");
        if (!tmp) { close(fd); return {}; }
        for (const auto& t : texts)
            fprintf(tmp, "%s\n", t.c_str());
        fclose(tmp);

        std::string cmd = udpipePath + " --tokenize --tag " + modelPath
                        + " < " + tmpfile + " 2>/dev/null";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) { unlink(tmpfile.c_str()); return {}; }

        std::vector<Token> tokens;
        char line[4096];
        while (fgets(line, sizeof(line), pipe)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            std::istringstream iss(line);
            std::string id, word, lemma, pos, rest;
            iss >> id >> word >> lemma >> pos;
            if (!id.empty() && id[0] != '.') {
                tokens.push_back({word, pos});
            }
        }
        pclose(pipe);
        unlink(tmpfile.c_str());
        return tokens;
    }

    static bool isNoun(const std::string& pos) {
        return pos == "NN" || pos == "NNS" || pos == "NNP" || pos == "NNPS";
    }

    static bool isVerb(const std::string& pos) {
        return pos == "VB" || pos == "VBD" || pos == "VBG" || pos == "VBN"
            || pos == "VBP" || pos == "VBZ";
    }

    static bool isAdj(const std::string& pos) {
        return pos == "JJ" || pos == "JJR" || pos == "JJS";
    }

    static bool isAdv(const std::string& pos) {
        return pos == "RB" || pos == "RBR" || pos == "RBS";
    }

private:
    std::string udpipePath;
    std::string modelPath;
};
