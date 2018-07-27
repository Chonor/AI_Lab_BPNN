#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>

using namespace std;
vector<string>str;
vector< vector<double> >Data;
vector< vector<double> >Out;
vector<double>cnt;
int  main() {
	string src;
	cin >> src;
	ifstream in;
	in.open(src);
	string tmp;
	while (getline(in, tmp))str.push_back(tmp);
	for (int s = 0; s<str.size(); s++) {
		tmp = str[s];
		vector<double>vtmp;
		string stmp = "";
		for (int i = 0; i < tmp.length(); i++) {//输入没什么说的
			if (tmp[i] != ',')stmp += tmp[i];
			else {
				stringstream ss(stmp);
				double a;
				ss >> a;
				vtmp.push_back(a);
				stmp = "";
			}
		}
		Data.push_back(vtmp);
		stringstream ss(stmp);
		double a;
		ss >> a;
		cnt.push_back(a);
	}
	for (int i = 0; i < str.size(); i++) {
		vector<double>vtmp;
		for (int j = 0; j < 44; j++)
			vtmp.push_back(0);
		Out.push_back(vtmp);
	}
	for (int i = 0; i < Data.size(); i++) {
		vector<double>vtmp=Data[i];
		Out[i][vtmp[0]-1] = 1;
		Out[i][vtmp[1] + 4] = 1;
		Out[i][28] = vtmp[2];
		Out[i][vtmp[3] + 29] = 1;
		Out[i][vtmp[4] + 35] = 1;
		Out[i][40] = vtmp[5];
		Out[i][41] = vtmp[6];
		Out[i][42] = vtmp[7];
		Out[i][43] = vtmp[8];
	}
	ofstream out;
	cin >> src;
	out.open(src+".txt");
	for (int i = 0; i < Out.size(); i++) {
		vector<double>vtmp = Out[i];
		for (int j = 0; j < Out[i].size(); j++) {
			out << vtmp[j] << ' ';
		}
		out << cnt[i] << endl;
	}
}
