#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <cmath>

using namespace std;


string readFileIntoString(char * filename){
    ifstream ifile(filename);
    ostringstream buf;
    char ch;
    while(buf&&ifile.get(ch))
        buf.put(ch);
    //返回与流对象buf关联的字符串
    return buf.str();
}

void split(const std::string& s,
    std::vector<std::string>& sv,
                   const char delim = ' ') {
    sv.clear();
    std::istringstream iss(s);
    std::string temp;

    while (std::getline(iss, temp, delim)) {
        sv.emplace_back(std::move(temp));
    }

    return;
}

vector<vector<double>> loadPos(){
    string loads = readFileIntoString("pos.txt");
    vector<string> out1;
    split(loads, out1, '\n');
    vector<vector<double>> pos;
    for(int i=0;i<out1.size();i++){
        vector<double> tmp;
        vector<string> out2;
        split(out1[i], out2, ' ');
        for(int j=0;j<out2.size();j++){
            tmp.push_back(stod(out2[j]));
        }
        if (tmp.size()>2)
            pos.push_back(tmp);
    }
    return pos;
}

vector<double> loadCharges(){
    string loads = readFileIntoString("charges.txt");
    vector<string> out1;
    split(loads, out1, '\n');
    vector<double> chrg;
    for(int i=0;i<out1.size();i++){
        chrg.push_back(stod(out1[i]));
    }
    return chrg;
}

vector<vector<int>> loadBlist(){
    string loads = readFileIntoString("blist.txt");
    vector<string> out1;
    split(loads, out1, '\n');
    vector<vector<int>> blist;
    for(int i=0;i<out1.size();i++){
        vector<int> tmp;
        vector<string> out2;
        split(out1[i], out2, ' ');
        for(int j=0;j<out2.size();j++){
            tmp.push_back(stoi(out2[j]));
        }
        if (tmp.size()>1)
            blist.push_back(tmp);
    }
    return blist;
}

vector<vector<int>> loadAlist(){
    string loads = readFileIntoString("alist.txt");
    vector<string> out1;
    split(loads, out1, '\n');
    vector<vector<int>> blist;
    for(int i=0;i<out1.size();i++){
        vector<int> tmp;
        vector<string> out2;
        split(out1[i], out2, ' ');
        for(int j=0;j<out2.size();j++){
            tmp.push_back(stoi(out2[j]));
        }
        if (tmp.size()>2)
            blist.push_back(tmp);
    }
    return blist;
}

vector<vector<double>> loadBparam(){
    string loads = readFileIntoString("bparam.txt");
    vector<string> out1;
    split(loads, out1, '\n');
    vector<vector<double>> blist;
    for(int i=0;i<out1.size();i++){
        vector<double> tmp;
        vector<string> out2;
        split(out1[i], out2, ' ');
        for(int j=0;j<out2.size();j++){
            tmp.push_back(stoi(out2[j]));
        }
        if (tmp.size()>1)
            blist.push_back(tmp);
    }
    return blist;
}

vector<vector<double>> loadAparam(){
    string loads = readFileIntoString("aparam.txt");
    vector<string> out1;
    split(loads, out1, '\n');
    vector<vector<double>> blist;
    for(int i=0;i<out1.size();i++){
        vector<double> tmp;
        vector<string> out2;
        split(out1[i], out2, ' ');
        for(int j=0;j<out2.size();j++){
            tmp.push_back(stoi(out2[j]));
        }
        if (tmp.size()>2)
            blist.push_back(tmp);
    }
    return blist;
}

vector<double> loadCell(){
    string loads = readFileIntoString("aparam.txt");
    vector<string> out1;
    split(loads, out1, '\n');
    vector<string> out2;
    split(out1[0], out2, ' ');
    vector<double> blist;
    for(int i=0;i<out2.size();i++){
        blist.push_back(stod(out2[i]));
    }
    return blist;
}

// CUDA算dQ/dQdx


// CUDA算Ewald

// 算距离
double distance(vector<double> p1, vector<double> p2, vector<double> cell){
    double dx = abs(p1[0] - p2[0]);
    double dy = abs(p1[1] - p2[1]);
    double dz = abs(p1[2] - p2[2]);
    if (dx > cell[0] / 2.0){
        dx -= cell[0] / 2.0;
    }
    if (dy > cell[1] / 2.0){
        dy -= cell[1] / 2.0;
    }
    if (dz > cell[2] / 2.0){
        dz -= cell[2] / 2.0;
    }
    return sqrt(dx*dx+dy*dy+dz*dz);
}

// CPU算Ewald
void calcEwaldRec(vector<vector<int>> klist, vector<vector<double>> pos, vector<double> charges, vector<double> cell){
    return;
}

int main(){
    vector<vector<double>> pos = loadPos();
    vector<double> charges = loadCharges();
    vector<vector<int>> bondlist = loadBlist();
    vector<vector<double>> bondparam = loadBparam();
    vector<vector<int>> anglelist = loadAlist();
    vector<vector<double>> angleparam = loadAparam();
    vector<double> cell = loadCell();
    cout << distance(pos[1], pos[2], cell) << endl;
    return 0;
}