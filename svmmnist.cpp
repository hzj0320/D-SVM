#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

using namespace std;

const string training_image_fn = "train-images.idx3-ubyte";
const string training_label_fn = "train-labels.idx1-ubyte";

const string testing_image_fn = "t10k-images.idx3-ubyte";
const string testing_label_fn = "t10k-labels.idx1-ubyte";

const int nTraining = 60000;
const int nTesting = 10000;
const int width = 28;
const int height = 28;
const int n1 = width * height; // = 784, without bias neuron 
const int n2 = n1+1;
const int n3 = 10;
const double svmparma = 5.0;
const int N = 50;
const int TN = 20;
const double J = 6;
const double Tres = 0.001;
const int step=15;
int index = 0;
int indext = 0;
ifstream image;
ifstream imaget;
ifstream label;
ifstream labelt;
ofstream report;
//int d1[width + 1][height + 1];
vector<double> d(n1);
vector<double> dd(n1);
vector<vector<double> > trainset(nTraining,vector<double>(n1));
vector<double> labelset(nTraining);
vector<vector<double> > testset(nTesting, vector<double>(n1));
vector<double> tlabelset(nTesting);


void Image() {
	// Reading image
	char number;
	int m = 0;
	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			image.read(&number, sizeof(char));
			if (number == 0) {
				d[m]=0;
				m++;
			}
			else {
				//cout << (int)number;
				d[m]=number;
				m++;
			}
		}
	}
	
	//cout << "Image:" << endl;
	/*for (int i = 0; i < m; i++) {
		cout << d[i];
	}*/

	label.read(&number, sizeof(char));
	if ((int)number == 2 || (int)number == 9) {
		for (int j = 0; j < n1; j++)
			trainset[index][j] = d[j];
		if (number == 2)
			labelset[index] = 1;
		if (number == 9)
			labelset[index] = -1;
		index++;
	}
}
void TImage() {
	// Reading image
	char number;
	int m = 0;
	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			imaget.read(&number, sizeof(char));
			if (number == 0) {
				dd[m] = 0;
				m++;
			}
			else {
				//cout << (int)number;
				dd[m] = number;
				m++;
			}
		}
	}

	//cout << "Image:" << endl;
	/*for (int i = 0; i < m; i++) {
		cout << d[i];
	}*/

	labelt.read(&number, sizeof(char));
	if ((int)number == 2 || (int)number == 9) {
		for (int j = 0; j < n1; j++)
			testset[indext][j] = dd[j];
		if (number == 2)
			tlabelset[indext] = 1;
		if (number == 9)
			tlabelset[indext] = -1;
		indext++;
	}
}
vector< vector<double> > matrix_add(vector< vector<double> > arrA, vector< vector<double> > arrB) {
	//矩阵arrA的行数  
	int rowA = arrA.size();
	//矩阵arrA的列数  
	int colA = arrA[0].size();
	vector< vector<double> >  res;
	res.resize(rowA);
	for (int i = 0; i < rowA; ++i)
	{
		res[i].resize(colA);
	}

	for (int i = 0; i < rowA; i++) {
		for (int j = 0; j < colA; j++) {
			res[i][j] = arrA[i][j] + arrB[i][j];
		}
	}
	return res;
}

vector< vector<double> > matrix_sub(vector< vector<double> > arrA, vector< vector<double> > arrB) {
	//矩阵arrA的行数  
	int rowA = arrA.size();
	//矩阵arrA的列数  
	int colA = arrA[0].size();
	vector< vector<double> >  res;
	res.resize(rowA);
	for (int i = 0; i < rowA; ++i)
	{
		res[i].resize(colA);
	}

	for (int i = 0; i < rowA; i++) {
		for (int j = 0; j < colA; j++) {
			res[i][j] = arrA[i][j] - arrB[i][j];
		}
	}
	return res;
}

vector< vector<double> > matrix_multiply(vector< vector<double> > arrA, vector< vector<double> > arrB)
{
	//矩阵arrA的行数  
	int rowA = arrA.size();
	//cout << rowA << endl;
	//矩阵arrA的列数  
	int colA = arrA[0].size();
	//cout << colA << endl;
	//矩阵arrB的行数  
	int rowB = arrB.size();
	//cout << rowB << endl;
	//矩阵arrB的列数  
	int colB = arrB[0].size();
	//cout << colB << endl;
	//相乘后的结果矩阵  
	vector< vector<double> >  res;
	double m;
	if (colA != rowB)//如果矩阵arrA的列数不等于矩阵arrB的行数。则返回空  
	{
		return res;
	}
	else
	{
		//设置结果矩阵的大小，初始化为为0  
		res.resize(rowA);
		for (int i = 0; i < rowA; ++i)
		{
			res[i].resize(colB);
		}
		//cout << res[0][0];
		//矩阵相乘  
		
		/*for (int i = 0; i < rowA; ++i)
		{
			for (int j = 0; j < colB; ++j)
			{
				for (int k = 0; k < colA; ++k)
				{
					res[i][j] += arrA[i][k] * arrB[k][j];
				}
			}
			
		}*/
		int i, k;
		for (i = 0; i < rowA; i++) {
			for (k = 0; k < colA; k++) {
				m= arrA[i][k];
				int j;
				for (j = 0; j < colB; j++) {
					res[i][j] += m * arrB[k][j];
				}
			}
		}
	}
	return res;
}
bool Gauss(vector< vector<double> > A, vector< vector<double> > B, int n)
{
	int i, j, k;
	double max, temp;
	vector< vector<double> > t(N, vector<double>(N));                //临时矩阵  
								  //将 A 矩阵存放在临时矩阵 t [ n ] [ n ] 中  
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			t[i][j] = A[i][j];
		}
	}
	//初始化 B 矩阵为单位阵  
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			B[i][j] = (i == j) ? (double)1 : 0;
		}
	}
	for (i = 0; i < n; i++)
	{
		//寻找主元  
		max = t[i][i];
		k = i;
		for (j = i + 1; j < n; j++)
		{
			if (fabs(t[j][i]) > fabs(max))
			{
				max = t[j][i];
				k = j;
			}
		}
		//如果主元所在行不是第 i 行，进行行交换  
		if (k != i)
		{
			for (j = 0; j < n; j++)
			{
				temp = t[i][j];
				t[i][j] = t[k][j];
				t[k][j] = temp;
				//B伴随交换  
				temp = B[i][j];
				B[i][j] = B[k][j];
				B[k][j] = temp;
			}
		}
		//判断主元是否为 0 , 若是, 则矩阵 A 不是满秩矩阵,不存在逆矩阵  
		if (t[i][i] == 0)
		{
			cout << "There is no inverse matrix!";
			return false;
		}
		//消去A的第i列除去i行以外的各行元素  
		temp = t[i][i];
		for (j = 0; j < n; j++)
		{
			t[i][j] = t[i][j] / temp;        //主对角线上的元素变为1  
			B[i][j] = B[i][j] / temp;        //伴随计算  
		}
		for (j = 0; j < n; j++)        //第 0 行 - > 第 n 行  
		{
			if (j != i)                //不是第 i 行  
			{
				temp = t[j][i];
				for (k = 0; k < n; k++)        //第 j 行元素  -  i 行元素 * j 列 i 行元素  
				{
					t[j][k] = t[j][k] - t[i][k] * temp;
					B[j][k] = B[j][k] - B[i][k] * temp;
				}
			}
		}
	}
	return true;
}
vector<vector<double> > initializeY(int start,int n) {
	vector<vector<double> > Y(N,vector<double>(N));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j)
				Y[i][j] = labelset[start];
			start++;
		}
	}
	return Y;
}
vector<vector<double> > initializeTY(int start, int n) {
	vector<vector<double> > Y(N, vector<double>(N));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j)
				Y[i][j] = tlabelset[start];
			start++;
		}
	}
	return Y;
}
vector<vector<double> > initializeX(int start, int n, int vsize) {
	vector<vector<double> > X(N,vector<double>(n2));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < vsize; j++) {
			if (j != n2 - 1)
				X[i][j] = trainset[start][j];
			else
				X[i][j] = 1;
		}
		start++;
	}
	return X;
}
vector<vector<double> > initializeTX(int start, int n, int vsize) {
	vector<vector<double> > X(N, vector<double>(n2));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < vsize; j++) {
			if (j != n2 - 1)
				X[i][j] = testset[start][j];
			else
				X[i][j] = 1;
		}
		start++;
	}
	return X;
}

vector<vector<double> > initializeVa(int n) {
	vector< vector<double> > v(n, vector<double>(1));
	srand(time(NULL));
	for (int i = 0; i < n; i++) {
		v[i][0] = rand() % 100 * 0.01;
	}
	return v;
}

vector<vector<double> > Tran(vector<vector<double> > X, int n, int vsize) {
	vector<vector<double> > TX(vsize, vector<double>(n));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < vsize; j++) {
			TX[j][i] = X[i][j];
		}
	}
	return TX;
}

vector<vector<double> > countQ(vector<vector<double> > matrix, vector <vector<double> > j) {
	vector< vector<double> > Q(N, vector<double>(1));
	for (int l = 0; l < N; l++) {
		Q[l][0] = j[l][0] - matrix[l][0];
	}
	return Q;
}

vector<vector<double> >countR(vector<vector<double> >v, vector<vector<double> >Tv,vector<vector<double> >sub,vector<vector<double> >Tj, vector<vector<double> >E, double J, double C) {
	vector<vector<double> > cR(1, vector<double>(1));
	vector<vector<double> > tmp1(1, vector<double>(1));
	vector<vector<double> > tmp2(1, vector<double>(1));
	tmp1 = matrix_multiply(Tv, matrix_multiply(sub, v));
	tmp1[0][0] /= 2;
	tmp2 = matrix_multiply(Tj, E);
	double JC = J * C;
	tmp2[0][0] /= JC;
	cR[0][0] = tmp1[0][0] + tmp2[0][0];
	return cR;
}

int main() {
	image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
	label.open(training_label_fn.c_str(), ios::in | ios::binary); // Binary label file
	imaget.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
	labelt.open(testing_label_fn.c_str(), ios::in | ios::binary); // Binary label file

	char number;
	srand((unsigned)time(0));
	for (int i = 1; i <= 16; ++i) {
		image.read(&number, sizeof(char));
	}
	for (int i = 1; i <= 8; ++i) {
		label.read(&number, sizeof(char));
	}
	for (int i = 1; i <= 16; ++i) {
		imaget.read(&number, sizeof(char));
	}
	for (int i = 1; i <= 8; ++i) {
		labelt.read(&number, sizeof(char));
	}
	for (int i = 1; i <= nTraining; i++) {
		Image();
	}
	cout << "success";
	for (int i = 1; i <= nTesting; i++) {
		TImage();
	}
	/*for (int i = 0; i < index; i++) {
		for (int j = 0; j < n1; j++) {
			cout << trainset[i][j];
		}
		cout << endl;
		cout << "Label:" << labelset[i];
		cout << endl;
	}*/
	//initialize

	vector< vector<double> > Y1 = initializeY(0, N);
	vector< vector<double> > Y2 = initializeY(N, N);
	vector< vector<double> > Y3 = initializeY(2 * N, N);
	vector< vector<double> > Y4 = initializeY(3 * N, N);
	vector< vector<double> > Y5 = initializeY(4 * N, N);
	vector< vector<double> > Y6 = initializeY(5 * N, N);

	vector< vector<double> > X1 = initializeX(0, N, n2);
	vector< vector<double> > X2 = initializeX(N, N, n2);
	vector< vector<double> > X3 = initializeX(2 * N, N, n2);
	vector< vector<double> > X4 = initializeX(3 * N, N, n2);
	vector< vector<double> > X5 = initializeX(4 * N, N, n2);
	vector< vector<double> > X6 = initializeX(5 * N, N, n2);
	vector< vector<double> > TX1 = Tran(X1, N, n2);
	vector< vector<double> > TX2 = Tran(X2, N, n2);
	vector< vector<double> > TX3 = Tran(X3, N, n2);
	vector< vector<double> > TX4 = Tran(X4, N, n2);
	vector< vector<double> > TX5 = Tran(X5, N, n2);
	vector< vector<double> > TX6 = Tran(X6, N, n2);

	cout << X1.size()<<endl;
	cout << X1[0].size() << endl;
	cout << TX1.size() << endl;
	cout << TX1[0].size() << endl;
	
	double B1 = 2;
	double B2 = 3;
	vector< vector<double> > v1 = initializeVa(n2);
	vector< vector<double> > v2 = initializeVa(n2);
	vector< vector<double> > v3 = initializeVa(n2);
	vector< vector<double> > v4 = initializeVa(n2);
	vector< vector<double> > v5 = initializeVa(n2);
	vector< vector<double> > v6 = initializeVa(n2);
	vector< vector<double> > Tv1 = Tran(v1, n2, 1);
	vector< vector<double> > Tv2 = Tran(v2, n2, 1);
	vector< vector<double> > Tv3 = Tran(v3, n2, 1);
	vector< vector<double> > Tv4 = Tran(v4, n2, 1);
	vector< vector<double> > Tv5 = Tran(v5, n2, 1);
	vector< vector<double> > Tv6 = Tran(v6, n2, 1);

	vector< vector<double> > a1 = initializeVa(n2);
	vector< vector<double> > a2 = initializeVa(n2);
	vector< vector<double> > a3 = initializeVa(n2);
	vector< vector<double> > a4 = initializeVa(n2);
	vector< vector<double> > a5 = initializeVa(n2);
	vector< vector<double> > a6 = initializeVa(n2);

	vector< vector<double> > lmultiplier1(N, vector<double>(1));
	vector< vector<double> > lmultiplier2(N, vector<double>(1));
	vector< vector<double> > lmultiplier3(N, vector<double>(1));
	vector< vector<double> > lmultiplier4(N, vector<double>(1));
	vector< vector<double> > lmultiplier5(N, vector<double>(1));
	vector< vector<double> > lmultiplier6(N, vector<double>(1));

	vector < vector<double> > j1(N, vector<double>(1, 1));
	vector<vector<double> > Tj1(1, vector<double>(N, 1));
	vector<vector<double> > E(N, vector<double>(1,0.5));//松弛变量
	double cost = 0.2;//惩罚因子

	vector< vector<double> > IP(n2,vector<double>(n2));
	vector< vector<double> > IIP(n2, vector<double>(n2));
	for (int i = 0; i < n2; i++) {
		for (int j = 0; j < n2; j++) {
			if (i == j)
				IP[i][j] = 1;
			else
				IP[i][j] = 0;
			if (i == n2 - 1 && j == n2 - 1)
				IIP[i][j] = 1;
			else
				IIP[i][j] = 0;
		}
	}
	vector<vector<double> > subIP = matrix_sub(IP, IIP);

	vector< vector<double> > U1(n2, vector<double>(n2));
	for (int i = 0; i < n2; i++) {
		for (int j = 0; j < n2; j++) {
			U1[i][j] = (1 + 2 * svmparma * B1) * IP[i][j] - IIP[i][j];
		}
	}
	

	vector< vector<double> > _U1(n2, vector<double>(n2));
	for (int i = 0; i < n2; i++) {
		for (int j = 0; j < n2; j++) {
			if (U1[i][j] != 0)
				_U1[i][j] = 1 / U1[i][j];
			else
				_U1[i][j] = 0;
		}
	}
	
	cout << "hello";
	vector< vector<double> > U2(n2, vector<double>(n2));
	for (int i = 0; i < n2; i++) {
		for (int j = 0; j < n2; j++) {
			U2[i][j] = (1 + 2 * svmparma * B2) * IP[i][j] - IIP[i][j];
		}
	}

	vector< vector<double> > _U2(n2, vector<double>(n2));
	for (int i = 0; i < n2; i++) {
		for (int j = 0; j < n2; j++) {
			if (U2[i][j] != 0)
				_U2[i][j] = 1 / U2[i][j];
			else
				_U2[i][j] = 0;
		}
	}
	
	vector< vector<double> > matrix11 = matrix_multiply(matrix_multiply(matrix_multiply(matrix_multiply(Y1, X1), _U1), TX1), Y1);
	
	vector< vector<double> > matrix21 = matrix_multiply(matrix_multiply(Y1, X1), _U1);

	vector< vector<double> > matrix31 = matrix_multiply(matrix_multiply(_U1, TX1), Y1);

	vector< vector<double> > matrix13 = matrix_multiply(matrix_multiply(matrix_multiply(matrix_multiply(Y3, X3), _U1), TX3), Y3);

	vector< vector<double> > matrix23 = matrix_multiply(matrix_multiply(Y3, X3), _U1);

	vector< vector<double> > matrix33 = matrix_multiply(matrix_multiply(_U1, TX3), Y3);

	vector< vector<double> > matrix15 = matrix_multiply(matrix_multiply(matrix_multiply(matrix_multiply(Y5, X5), _U1), TX5), Y5);

	vector< vector<double> > matrix25 = matrix_multiply(matrix_multiply(Y5, X5), _U1);

	vector< vector<double> > matrix35 = matrix_multiply(matrix_multiply(_U1, TX5), Y5);

	vector< vector<double> > matrix16 = matrix_multiply(matrix_multiply(matrix_multiply(matrix_multiply(Y6, X6), _U1), TX6), Y6);

	vector< vector<double> > matrix26 = matrix_multiply(matrix_multiply(Y6, X6), _U1);

	vector< vector<double> > matrix36 = matrix_multiply(matrix_multiply(_U1, TX6), Y6);


	vector< vector<double> > matrix12 = matrix_multiply(matrix_multiply(matrix_multiply(matrix_multiply(Y2, X2), _U2), TX2), Y2);

	vector< vector<double> > matrix22 = matrix_multiply(matrix_multiply(Y2, X2), _U2);

	vector< vector<double> > matrix32 = matrix_multiply(matrix_multiply(_U2, TX2), Y2);

	vector< vector<double> > matrix14 = matrix_multiply(matrix_multiply(matrix_multiply(matrix_multiply(Y4, X4), _U2), TX4), Y4);

	vector< vector<double> > matrix24 = matrix_multiply(matrix_multiply(Y4, X4), _U2);

	vector< vector<double> > matrix34 = matrix_multiply(matrix_multiply(_U2, TX4), Y4);

	cout << "success" << endl;
	vector< vector<double> > _matrix11(N, vector<double>(N));
	vector< vector<double> > _matrix12(N, vector<double>(N));
	vector< vector<double> > _matrix13(N, vector<double>(N));
	vector< vector<double> > _matrix14(N, vector<double>(N));
	vector< vector<double> > _matrix15(N, vector<double>(N));
	vector< vector<double> > _matrix16(N, vector<double>(N));
	cout << "success3" << endl;
	if (Gauss(matrix11, _matrix11, N))
		cout << "success4" << endl;
	if (Gauss(matrix12, _matrix12, N))
		cout << "success4" << endl;
	if (Gauss(matrix13, _matrix13, N))
		cout << "success4" << endl;
	if (Gauss(matrix14, _matrix14, N))
		cout << "success4" << endl;
	if (Gauss(matrix15, _matrix16, N))
		cout << "success4" << endl;
	if (Gauss(matrix16, _matrix16, N))
		cout << "success4" << endl;

	vector<vector<double> > pR1 = countR(v1, Tv1, subIP, Tj1, E, J, cost);
	vector<vector<double> > R1(1, vector<double>(1));
	
	int flag = false;
	int m = 0;

	//train
	while(m<=400){
		vector< vector<double> > f1(n2,vector<double>(1));
		vector< vector<double> > f2(n2, vector<double>(1));
		vector< vector<double> > f3(n2, vector<double>(1));
		vector< vector<double> > f4(n2, vector<double>(1));
		vector< vector<double> > f5(n2, vector<double>(1));
		vector< vector<double> > f6(n2, vector<double>(1));
		
		for (int l = 0; l < n2; l++) {
			f1[l][0] = (-2) * a1[l][0] + svmparma * (v1[l][0] + v3[l][0] + v1[l][0] + v2[l][0]);
			f3[l][0] = (-2) * a3[l][0] + svmparma * (v3[l][0] + v1[l][0] + v3[l][0] + v2[l][0]);
			f2[l][0] = (-2) * a2[l][0] + svmparma * (v2[l][0] + v1[l][0] + v2[l][0] + v3[l][0]+ v2[l][0]+ v4[l][0]);
			f4[l][0] = (-2) * a4[l][0] + svmparma * (v4[l][0] + v4[l][0] + v4[l][0] + v6[l][0] + v2[l][0] + v4[l][0]);
			f5[l][0] = (-2) * a5[l][0] + svmparma * (v5[l][0] + v6[l][0] + v5[l][0] + v4[l][0]);
			f6[l][0] = (-2) * a6[l][0] + svmparma * (v6[l][0] + v5[l][0] + v6[l][0] + v4[l][0]);
		}
		vector< vector<double> > matrix41 = matrix_multiply(matrix21, f1);
		vector< vector<double> > matrix42 = matrix_multiply(matrix22, f2);
		vector< vector<double> > matrix43 = matrix_multiply(matrix23, f3);
		vector< vector<double> > matrix44 = matrix_multiply(matrix24, f4);
		vector< vector<double> > matrix45 = matrix_multiply(matrix25, f5);
		vector< vector<double> > matrix46 = matrix_multiply(matrix26, f6);
		vector< vector<double> > q1 = countQ(matrix41, j1);
		vector< vector<double> > q2 = countQ(matrix42, j1);
		vector< vector<double> > q3 = countQ(matrix43, j1);
		vector< vector<double> > q4 = countQ(matrix44, j1);
		vector< vector<double> > q5 = countQ(matrix45, j1);
		vector< vector<double> > q6 = countQ(matrix46, j1);
		

		lmultiplier1 = matrix_multiply(_matrix11, q1);
		lmultiplier2 = matrix_multiply(_matrix12, q2);
		lmultiplier3 = matrix_multiply(_matrix13, q3);
		lmultiplier4 = matrix_multiply(_matrix14, q4);
		lmultiplier5 = matrix_multiply(_matrix15, q5);
		lmultiplier6 = matrix_multiply(_matrix16, q6);

		
		v1 = matrix_add(matrix_multiply(matrix31, lmultiplier1), matrix_multiply(_U1, f1));
		v3 = matrix_add(matrix_multiply(matrix33, lmultiplier3), matrix_multiply(_U1, f3));
		v2 = matrix_add(matrix_multiply(matrix32, lmultiplier2), matrix_multiply(_U2, f2));
		v4 = matrix_add(matrix_multiply(matrix34, lmultiplier4), matrix_multiply(_U2, f4));
		v5 = matrix_add(matrix_multiply(matrix35, lmultiplier5), matrix_multiply(_U1, f5));
		v6 = matrix_add(matrix_multiply(matrix36, lmultiplier6), matrix_multiply(_U1, f6));
		
		for (int l = 0; l < n2; l++) {
			a1[l][0] = a1[l][0] + (1 / 2) * svmparma * (v1[l][0] - v3[l][0] + v1[l][0] - v2[1][0]);
			a3[l][0] = a3[l][0] + (1 / 2) * svmparma * (v3[l][0] - v1[l][0] + v3[l][0] - v2[1][0]);
			a2[l][0] = a2[l][0] + (1 / 2) * svmparma * (v2[l][0] - v1[l][0] + v2[l][0] - v3[1][0] + v2[l][0] - v4[l][0]);
			a4[l][0] = a4[l][0] + (1 / 2) * svmparma * (v4[l][0] - v5[l][0] + v4[l][0] - v6[1][0] + v4[l][0] - v2[l][0]);
			a5[l][0] = a5[l][0] + (1 / 2) * svmparma * (v5[l][0] - v6[l][0] + v5[l][0] - v4[1][0]);
			a6[l][0] = a6[l][0] + (1 / 2) * svmparma * (v6[l][0] - v5[l][0] + v6[l][0] - v4[1][0]);	
		}
		R1 = countR(v1, Tv1, subIP, Tj1, E, J, cost);
		double stop = abs(R1[0][0] - pR1[0][0]);
		cout << stop << endl;
		/*if (abs(R1[0][0] - pR1[0][0]) < Tres) {
			flag = true;
			break;
		}*/
		cout << ++m << endl;
		pR1 = R1;
	}
	cout << "success" << endl;
	for (int i = 0; i < n2; i++) {
		cout << v1[i][0] << endl;
	}

	//test
	vector<vector<double> > TestX = initializeTX(0, TN, n2);
	vector<vector<double> > TestY = initializeTY(0, TN);
	vector<vector<double> > G(TN, vector<double>(1));
	double cnt = 0;
	G = matrix_multiply(TestX, v1);
	for (int i = 0; i < TN; i++) {
		if ((G[i][0] <= -0.5 && TestY[i][i] <= -0.5) || (G[i][0] >= 0.5 && TestY[i][i] >= 0.5)) {
			cnt++;
			cout << "right" << endl;
		}
	}
	cout << cnt / TN << endl;

	return 0;
}


