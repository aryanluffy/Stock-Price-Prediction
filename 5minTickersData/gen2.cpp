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

map<string,vector<double>> generateParameters(string csvfile){
    map <string,vector<double>> toret;
    ifstream in(csvfile+".csv");
    ofstream out(csvfile+"parameters.csv");
    string x,y;
    in>>x;
    vector <double> data[5];
    vector <string> dates;
    while(in>>x){
        in>>y;
        string curr;
        vector <float> temp;
        for(int i=15;i<y.size();i++){
            if(y[i]==','){
                temp.push_back(stod(curr));
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
    vector <double> low(data[0].size(),0),high(data[0].size(),0),close(data[0].size(),0),rsi(data[0].size(),0);
    vector <double> bollingerBand[2];
    //0 is lower band and 1 is upper band
    bollingerBand[0].assign(low.size(),0);
    bollingerBand[1].assign(low.size(),0);
    vector <double> ema26(data[0].size(),0),ema12(data[0].size(),0),cci(data[0].size(),0);
    for(int i=0;i<12;i++)ema12[11]+=data[3][i];
    for(int i=0;i<26;i++)ema26[25]+=data[3][i];
    ema12[11]/=12;
    ema26[25]/26;
    for(int i=12;i<data[0].size();i++)
        ema12[i]=data[3][i]*(2.0/13)+(11.0/13)*ema12[i-1];
    for(int i=26;i<data[0].size();i++)
        ema26[i]=data[3][i]*(2.0/26)+(24.0/26)*ema26[i-1];
    vector <double> diff(data[0].size(),0);
    for(int  i=0;i<data[0].size();i++)diff[i]=ema26[i]-ema12[i];
    vector <double> macd(data[0].size(),0);
    for(int i=0;i<9;i++)macd[8]+=diff[i];
    macd[8]/=9;
    for(int i=9;i<data[0].size();i++)macd[i]=diff[i]*(2.0/10)+(0.8)*macd[i-1];
    double minval=0,maxval=0,closeval=0;
    for(int i=13;i<data[0].size();i++){
        float gains=0;
        float losses=0;
        for(int j=i-12;j<=i;j++){
            gains+=max(0.0,data[3][j]-data[3][j-1]);
            losses+=max(0.0,data[3][j-1]-data[3][j]);
        }
        if(gains+losses!=0.0)
        rsi[i]=gains/(gains+losses);
    }
    vector <double> matypical20(data[0].size());
    for(int i=0;i<20;i++)matypical20[19]+=(data[1][i]+data[2][i]+data[3][i])/3;
    for(int i=20;i<data[0].size();i++){
        matypical20[i]=matypical20[i-1]+(data[1][i]+data[2][i]+data[3][i])/3-(data[1][i-20]+data[2][i-20]+data[3][i-20])/3;
    }
    for(int i=0;i<matypical20.size();i++)matypical20[i]/=20;
    for(int i=20;i<cci.size();i++){
        double sum=0;
        double sum2=0;
        for(int j=i-19;j<=i;j++){
            sum+=abs((data[1][j]+data[2][j]+data[3][j])/3-matypical20[i]);
            sum2+=abs((data[1][j]+data[2][j]+data[3][j])/3-matypical20[i])*abs((data[1][j]+data[2][j]+data[3][j])/3-matypical20[i]);
        }
        sum/=20;
        sum2/=20;
        if(sum)
        cci[i]=((data[1][i]+data[2][i]+data[3][i])/3-matypical20[i])/(sum);
        else 
        cci[i]=1000000000;
        bollingerBand[0][i]=matypical20[i]-2*sqrt(sum2);
        bollingerBand[1][i]=matypical20[i]+2*sqrt(sum2);
    }
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
    int prev[3][1][data[0].size()];
    memset(prev,0,sizeof prev);
    for(int i=0;i<data[0].size();i++){
        prev[0][0][i]=low[i];
        prev[1][0][i]=high[i];
        prev[2][0][i]=close[i];
    }
    out<<"Date,TimeOfday,Open,High,Low,Close,Volume,";
    out<<"RSI,MACD,CCI,BollingerBandL,BollingerBandU,Month,DayOfWeek,DayOfMonth\n";
    for(int i=0;i<dates.size();i++){
        out<<dates[i]<<",";
        out<<dateTimeToMinuteTime(dates[i])<<",";
        for(int j=0;j<5;j++){
            out<<data[j][i]<<",";
            toret[dates[i]].push_back(data[j][i]);
        }    
        int d,m,y;
        d=stoi(dates[i].substr(8,2));
        m=stoi(dates[i].substr(5,2));
        y=stoi(dates[i].substr(0,4));
        int dow=dayofweek(d,m,y);
        out<<rsi[i]<<","<<macd[i]<<","<<cci[i]<<","<<bollingerBand[0][i]<<","<<bollingerBand[1][i]<<",";
        out<<dates[i].substr(5,2)<<",";
        out<<dow<<","<<d<<"\n";
        toret[dates[i]].push_back(rsi[i]);
        toret[dates[i]].push_back(macd[i]);
        toret[dates[i]].push_back(cci[i]);
        toret[dates[i]].push_back(bollingerBand[0][i]);
        toret[dates[i]].push_back(bollingerBand[1][i]);
        toret[dates[i]].push_back(stoi(dates[i].substr(5,2)));
        toret[dates[i]].push_back(dow);
        toret[dates[i]].push_back(d);
    }
    return toret;
}

int main(){
    string tickers[]={
        "AIRTEL","AXISBANK","BAJAJFINSV",\
        "BAJFINANCE","CHOLAFIN","HDFCBANK",\
        "ICICIBANK","KOTAKBANK","L&T",\
        "M&MFIN","REIANCE","SBIN","WIPRO"
    };
    ofstream out("ISSE_NAHI_HUA_KUCH_BHI_TOH_CHODO");
    vector <map<string,vector <double>>> mp;
    for(int i=0;i<13;i++){
        mp.push_back(generateParameters(tickers[i]));
    }
    out<<"date,";
    for(int i=0;i<133;i++){
        out<<"feature_"+to_string(i+1)<<",";
    }
    out<<"feature_134\n";
    vector <double> scale_down(13,1);
    map <string,string> monthbegin;
    for(auto x:mp[3]){
        if(monthbegin.find(x.first.substr(0,7))==monthbegin.end())
        monthbegin[x.first.substr(0,7)]=x.first;
    }
    for(auto x:mp[3]){
        bool flag=true;
        for(int i=0;i<13;i++)if(mp[i].find(x.first)==mp[i].end())flag=false;
        if(flag==0)continue;
        if(monthbegin[x.first.substr(0,7)]==x.first){
            for(int i=0;i<13;i++)scale_down[i]=mp[i][x.first][0];
        }
        out<<x.first<<",";
        for(int i=0;i<13;i++){
            for(int j=0;j<10;j++){
                if(j==4 || j==5 || j==7)
                out<<mp[i][x.first][j]<<",";
                else out<<mp[i][x.first][j]/scale_down[i]<<",";
            }
        }
        out<<dateTimeToMinuteTime(x.first)<<","<<mp[0][x.first][10]<<","<<mp[0][x.first][11]<<","<<mp[0][x.first][12]<<"\n";
    }
    return 0;
}
