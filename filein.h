// Read in a delimited file
#ifndef FILEIN
#define FILEIN

#include <string>

using namespace std;

void split(const string& s, char c, vector<string>& v) {
   string::size_type i = 0;
   string::size_type j = s.find(c);

   while (j != string::npos) {
      v.push_back(s.substr(i, j-i));
      i = ++j;
      j = s.find(c, j);

      if (j == string::npos)
         v.push_back(s.substr(i, s.length( )));
   }
}

void loadCSV(ifstream& in, vector<string>& data)
{
  string tmp;
  while(!in.eof( ))
  {
        getline(in, tmp, '\n');
        data.push_back(tmp);
        tmp.clear();
  }
}
        
   
void loadCSV(ifstream& in, vector<vector<string>* >& data) {

   vector<string>* p = NULL;
   string tmp;
   while (!in.eof( )) 
   {
      getline(in, tmp, '\n');                     // Grab the next line
      p = new vector<string> ( );
      split(tmp, ',', *p);                        
      data.push_back(p);
      //cout << tmp << '\n';
      tmp.clear( );
 }
}

long int hex2int(const string& hexStr) {
   char *offset;
   if (hexStr.length( ) > 2) {
      if (hexStr[0] == '0' && hexStr[1] == 'x') {
         return strtol(hexStr.c_str( ), &offset, 0);
      }
   }
   return strtol(hexStr.c_str( ), &offset, 16);
}

long double hex2d(const string& hexStr) {
   char *offset;
   if (hexStr.length( ) > 2) {
      if (hexStr[0] == '0' && hexStr[1] == 'x') {
         return strtod(hexStr.c_str( ), &offset);
      }
   }
   return strtod(hexStr.c_str( ), &offset);
}

#endif

