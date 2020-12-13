#include <bits/stdc++.h>
#include <fstream>
using namespace std;

int dayofweek(int d, int m, int y)  
{  
    static int t[] = { 0, 3, 2, 5, 0, 3, 
                       5, 1, 4, 6, 2, 4 };  
    y -= m < 3;  
    return ( y + y / 4 - y / 100 +  
             y / 400 + t[m - 1] + d) % 7;  
}
int dateTimeToMinuteTime(string date){
    return (10*(date[11]-'0')+(date[12]-'0'))*60+(10*(date[14]-'0')+(date[15]-'0'));
}

int main(){
    string csvfile;
    cin>>csvfile;
    ifstream in(csvfile+".csv");
    ofstream out(csvfile+"parameters.csv");
    string x,y;
    in>>x;
    vector <float> data[5];
    vector <string> dates;
    while(in>>x){
        in>>y;
        string curr;
        vector <float> temp;
        for(int i=15;i<y.size();i++){
            if(y[i]==','){
                temp.push_back(stof(curr));
                curr="";
            }
            else{
                curr.push_back(y[i]);
            }
        }
        temp.push_back(stof(curr));
        for(int i=0;i<5;i++)data[i].push_back(temp[i]);
        dates.push_back(x+" "+y.substr(0,14));
    }
    vector <float> low(data[0].size(),0),high(data[0].size(),0),close(data[0].size(),0);
    float minval=0,maxval=0,closeval=0;
    for(int i=0;i<data[0].size();i++){
        // cout<<dates[i].substr(11)<<"\n";
        if(dates[i].substr(11)=="09:15:00+05:30"){
            // cout<<"OK\n";
            low[i]=minval;
            high[i]=maxval;
            close[i]=closeval;
            minval=data[2][i];
            closeval=data[3][i];
            maxval=data[1][i];
        }
        else{
            if(i){
                low[i]=low[i-1];
                high[i]=high[i-1];
                close[i]=close[i-1];
            }
            minval=min(minval,data[2][i]);
            maxval=max(maxval,data[1][i]);
            closeval=data[3][i];
        }
    }
    int prev[3][14][data[0].size()];
    memset(prev,0,sizeof prev);
    for(int i=0;i<data[0].size();i++){
        prev[0][0][i]=low[i];
        prev[1][0][i]=high[i];
        prev[2][0][i]=close[i];
    }
    for(int i=0;i<data[0].size();i++){
        for(int j=1;j<14;j++)
            for(int k=0;k<3;k++)
                if(i>74)
                prev[k][j][i]=prev[k][j-1][i-75];
    }
    out<<"Date,TimeOfday,Open,High,Low,Close,Volume,";
    for(int i=0;i<14;i++){
        for(int j=0;j<3;j++){
            if(j==0)out<<"Low"+to_string(i)<<",";
            else if(j==1)out<<"High"+to_string(i)<<",";
            else out<<"Close"+to_string(i)<<",";
        }
    }
    out<<"DayOfWeek,DayOfMonth\n";
    for(int i=0;i<dates.size();i++){
        out<<dates[i]<<",";
        out<<dateTimeToMinuteTime(dates[i])<<",";
        for(int j=0;j<5;j++)    
            out<<data[j][i]<<",";
        for(int k=0;k<14;k++)
            for(int l=0;l<3;l++)
                out<<prev[l][k][i]<<",";
        int d,m,y;
        d=stoi(dates[i].substr(8,2));
        m=stoi(dates[i].substr(5,2));
        y=stoi(dates[i].substr(0,4));
        int dow=dayofweek(d,m,y);
        out<<dow<<","<<d<<"\n";
    }
    return 0;
}
