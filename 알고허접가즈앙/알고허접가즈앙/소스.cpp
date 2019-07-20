#include<iostream>
#include<stdio.h>
#include<string>
#include<vector>
#include<algorithm>
#include<cstring>

using namespace std;

//Main Form
#if 0
int main(void)
{
	int test_case;
	cin >> test_case;

	for (int i = 0; i < test_case; i++)
	{

	}
	return 0;
}
#endif
//Dynamic Programing
#if 0
int tNum, row, col;
int board[21][21];

//����Ž�� : ������ ����.
/*
��������
1) input �κп��� error check �κ��� ��������.
�ع�
1) ���� Ž���� ����ؼ�, �� Point ���� ����
2) �ߺ��� ���, ���������� ������ �Ʒ��� ������ �����ؼ� �ߺ� ����.
3) ���� ���� ������ ������ ���� �������� ��Ÿ���� �ϳ��� �Լ��� ����.
*/
//input       
int input() {

	int count;
	register int i, j;
	char in;

	scanf("%d %d", &row, &col);
	count = 0;

	if (row < 1 || row > 20 || col < 1 || col > 20) return 1;

	for (i = 0; i < row; i++) {
		for (j = 0; j < col; j++) {
			scanf(" %c", &in);
			if (in == '.') {
				board[i][j] = 0;
				count++;
			}
			else board[i][j] = 1;
		}
	}
	count = count % 3;
	return count; //������� ���� 3�� ����� �ƴϸ� fail.
}
//������ ����.
const int coverType[4][3][2]{
	{{0,0}, {1,0}, {0,1}},
	{{0,0}, {0,1}, {1,1}},
	{{0,0}, {1,0}, {1,1}},
	{{0,0}, {1,0}, {1,-1}}
};
int set(int type, int rx, int cy, int cover) {

	int ok = 1;

	for (int i = 0; i < 3; i++) {
		int tx = rx + coverType[type][i][0];
		int ty = cy + coverType[type][i][1];

		if (tx < 0 || tx >= row || ty < 0 || ty >= col) ok = 0; //���� �˻�.
		else if ((board[tx][ty] += cover) > 1) ok = 0;
	}
	return ok;
}
int bruteForce() {

	int x = -1, y = -1;

	for (register int i = 0; i < row; i++) {
		for (register int j = 0; j < col; j++) {
			if (board[i][j] == 0) {
				x = i;
				y = j;
				break;
			}
		}
		if (y != -1) break;
	} //�����ؾ��� �κ� ã��.

	if (y == -1) return 1; //��ã������ �� ���� �Ŷ� return.
	int ret = 0;

	for (int type = 0; type < 4; type++) {
		if (set(type, x, y, 1)) ret += bruteForce();
		set(type, x, y, -1);
	}
	return ret;
}
int main() {

	int result;
	scanf("%d", &tNum);

	for (int i = 0; i < tNum; i++) {
		result = 0;
		if (input() != 0) {
			printf("%d\n", result);
		}
		else {
			result = bruteForce();
			printf("%d\n", result);
		}
	}
	return 0;
}
#endif //����Ž�� : ������ ����

#if 0
//����Ž�� : �ð� ���߱�.
/*
��������
1) ������ ���� ������ ��, �������κ��� ��Ʈ �� ���.
�ع�
1) ���� Ž���� ����ؼ�, �� ����ġ�� ���� ��� ��� Ȯ��.
2) ������ ��� ���⿡, 0���� 9������ ����.
*/
#define INF 999999999

int tNum;
int clocks[16];
int swtch;
int linked[10][5] = {
{0,1,2,-1,-1},
{3,7,9,11,-1},
{4,10,14,15,-1},
{0,4,5,6,7},
{6,7,8,10,12},
{0,2,14,15,-1},
{3,14,15,-1,-1},
{4,5,7,14,15},
{1,2,3,4,5},
{3,4,5,9,13}
};
void inputData() {
	for (register int i = 0; i < 16; i++) {
		scanf("%d", &clocks[i]);
	}
	return;
}
int checkAligned() {
	int cnt = 0;
	for (int i = 0; i < 16; i++) {
		if (clocks[i] == 12) cnt++;
	}
	return cnt == 16 ? 1 : 0;
}
void push(int swtch) {
	for (register int i = 0; i < 5; i++) {
		if (linked[swtch][i] != -1) {
			int clock = linked[swtch][i];
			clocks[clock] += 3;
			if (clocks[clock] == 15) clocks[clock] = 3;
		}
	}
}
int min(int a, int b) {
	return a < b ? a : b;
}
int solve(int swtch) {

	if (swtch == 10) {
		return checkAligned() == 1 ? 0 : INF;
	}//Ż�� ����.
	int ret = INF;
	for (int cnt = 0; cnt < 4; cnt++) {
		ret = min(ret, cnt + solve(swtch + 1));
		push(swtch);
	}
	return ret;
}
int main() {

	scanf("%d", &tNum);

	for (int i = 0; i < tNum; i++) {
		inputData();
		swtch = 0;
		int output = solve(0);

		if (output == INF) cout << -1 << endl;
		else cout << output << endl;
	}

	return 0;
}
#endif //����Ž�� : �ð� ���߱�.

#if 0
//����Ž�� + DP : JUMP GAME.
/*
��������
�ع�
1) ���� Ž���� ����Ѵ�.
2) Ư�� Input�� ���� �׻� ���� �����Ƿ�, DP�� Ȱ���Ѵ�.
*/

int tNum;
int ary[100][100];
int cache[100][100];
int width;

void inputData() {

	scanf("%d", &width);
	for (int row = 0; row < width; row++) {
		for (int col = 0; col < width; col++) {
			scanf("%d", &ary[row][col]);
			cache[row][col] = -1;
		}
	}
	return;
}
int jump(int curRow, int curCol) {

	if (curRow >= width || curCol >= width) return 0;

	if (curRow == width - 1 && curCol == width - 1) return 1;

	int& ret = cache[curRow][curCol];
	if (ret != -1) return ret;

	return ret = jump(curRow + ary[curRow][curCol], curCol) || jump(curRow, curCol + ary[curRow][curCol]);
}
int main() {

	scanf("%d", &tNum);

	for (int i = 0; i < tNum; i++) {
		inputData();

		int result = jump(0, 0);

		if (result) cout << "YES" << endl;
		else cout << "NO" << endl;
	}
	return 0;
}


#endif //����Ž�� + DP : JUMP GAME

#if 0
//����Ž�� + DP : WILD Pattern
/*
��������
1) �� �κй����� ��� ������, � ������ ��ͷ� ȣ�������� ����.
2) Vecotr�� ���� �ʱ�ȭ�� �ȵǼ� ������ Ʋ����, �ʱ�ȭ �Ű澲��.
�ع�
1) ���� Ž���� ����Ѵ�('*'�� ���, ���� Ž���� ����ϴµ�, �� �κй����� ��� �������� �߿�).
2) Ư�� Input�� ���� Ư�� �κй����� ���� ������� �����Ƿ�, DP�� Ȱ���Ѵ�.
*/
int testCase;
int patternNum;
int cache[101][101];
string WildPattern;
string inputString;
vector<string> outCandiate;

bool match(int patternOffset, int candidateOffset) {

	int& ret = cache[patternOffset][candidateOffset];
	if (ret != -1) return ret;

	while (patternOffset < WildPattern.size() && candidateOffset < inputString.size()
		&& (WildPattern[patternOffset] == inputString[candidateOffset] || WildPattern[patternOffset] == '?')) {
		patternOffset++;
		candidateOffset++;
	}

	if (patternOffset == WildPattern.size()) //pattern index�� ������ ����,
		return ret = (candidateOffset == inputString.size()); //candiate�� ������ ����! -> ��ġ.

	if (WildPattern[patternOffset] == '*') { //1)���߿� '*'�� ���°��;
		for (int skipCount = 0; candidateOffset + skipCount <= inputString.size(); skipCount++) {
			if (match(patternOffset + 1, candidateOffset + skipCount)) {
				return ret = 1;
			}
		}
	}
	return ret = 0; //�߰��� pattern�� �� ������.
}
void initial() {
	for (register int i = 0; i < 101; i++) {
		for (register int j = 0; j < 101; j++)
			cache[i][j] = -1;
	}
}
int main() {

	scanf("%d", &testCase);

	for (int i = 0; i < testCase; i++) {
		cin >> WildPattern;
		scanf("%d", &patternNum);
		for (int j = 0; j < patternNum; j++) {
			cin >> inputString;
			initial();
			if (match(0, 0)) {
				outCandiate.push_back(inputString);
			}
		}
		//�ĺ��� ���� �� ���.
		sort(outCandiate.begin(), outCandiate.end());
		for (int j = 0; j < outCandiate.size(); j++) {
			cout << outCandiate[j] << endl;
		}
		outCandiate.clear();
	}
	return 0;
}
#endif //����Ž�� + DP : WILD Pattern

#if 0
//����Ž�� + DP : TRIANGLEPATH
/*
��������
1) �ð� ���⵵ ����� ���� ����.
2) DP�� �������� ���� �Ŷ�� ��Ȳ�� ���������� ������ �ٲ㼭 DP�� �����Ű���� ��� ����
�ع�
1) �⺻������ ���� Ž���� ����Ѵ� -> ����Ž�� ����ϸ� �ð��ʰ�
2) �ð� �ȿ� ������ ���� �Է�(���)���� �ٿ�����.
3) ������ �κ� ������ �ᱹ, Ư�� �������� ������ ������ �ִ� ���� Path.
4) ���� ĳ�� �س��� ���

���Ӱ� �˰� �Ȱ�.
1) ���� �κ� ���� : ������ �����ظ� ���ϴ� �� �־�, �� �κ� �������� �� �����ص��� �� ��ü�� �����ذ� �Ǵ°�.
�ȵǴ� ��� : Ư�� �������� ����, �� �κ� ������ �����ذ� ������ ���ǿ� ����Ǵ� ���(���� Ư�� ���� �Ѵ´ٵ簡??)
*/

int testCount, triangleSize, rowCount;
int triangle[100][100];
int cache[100][100];

int max(int left, int right) {
	return left > right ? left : right;
}
int calcMaxPath(int row, int col) {

	if (row == triangleSize - 1) return triangle[row][col];

	int& cValue = cache[row][col];
	if (cValue != -1) return cValue;

	return cValue = triangle[row][col] + max(calcMaxPath(row + 1, col), calcMaxPath(row + 1, col + 1));
}
void inititalize() {
	for (register int i = 0; i < triangleSize; i++) {
		for (register int j = 0; j < triangleSize; j++) {
			cache[i][j] = -1;
			triangle[i][j] = 0;
		}
	}
}
int main() {
	int result;

	scanf("%d", &testCount);
	for (register int i = 0; i < testCount; i++) {
		scanf("%d", &triangleSize);
		inititalize();
		rowCount = 1;
		for (register int i = 0; i < triangleSize; i++) {
			for (register int j = 0; j < rowCount; j++) {
				scanf("%d", &triangle[i][j]);
			}
			rowCount++;
		}
		result = calcMaxPath(0, 0);
		printf("%d\n", result);
	}
}
#endif //����Ž�� + DP : TRIANGLEPATH

#if 0
//����Ž�� + DP : LIS
/*
��������

�ع�
1) �⺻������ ���� Ž���� ����Ѵ� -> ����Ž�� ����ϸ� �ð��ʰ�
2) ���� ������ �������� �������� ���� ���Ѵٴ� �ǹ̿��� �� ������ �������� ����.
-> �׷����� ������ ���� ���� ����� ���� �ö��� ���� ���ϴ� ���� ���߿� ����� �ʿ䰡 ����.
3) ���� ĳ�� �س��� ���

���Ӱ� �˰� �Ȱ�.
*/
int testNum;
int numArray;
int arr[501];
int cache[501];

void InputData() {
	scanf("%d", &numArray);

	for (int numCount = 0; numCount < numArray; numCount++) {
		cin >> arr[numCount];
		cache[numCount] = -1;
	}
}
int max(int x, int y) {
	return x > y ? x : y;
}
int lis(int start) {

	int& ret = cache[start];

	if (ret != -1) return ret;

	ret = 1;
	for (int next = start + 1; next < numArray; next++) {
		if (arr[start] < arr[next]) {
			ret = max(ret, lis(next) + 1);
		}
	}
	return ret;
}
int  main() {

	int maxLength;
	scanf("%d", &testNum);

	for (register int testCount = 0; testCount < testNum; testCount++) {
		InputData();
		maxLength = 0;
		for (int begin = 0; begin < numArray; begin++) {
			maxLength = max(maxLength, lis(begin));
		}
		printf("%d\n", maxLength);
	}

}
#endif //����Ž�� + DP : LIS

#if 0
//����Ž�� + DP : JLIS
/*
��������
1. ��ȭ�� ����°� ���� ����.
�ع�
1. �� �ٿ� ���� LIS�� ���ϰ� �� ��������, �ٸ� ������ ���� �� ���� �� �ִ��� Ȯ��.
���Ӱ� �˰� �Ȱ�.
-> long long NEGINF = numeric_limits<long long>::min();
*/
int testNum;
int aLength, bLength;
int a[100], b[100];
int cache[101][101];
long long NEGINF = numeric_limits<long long>::min();

void InputData() {
	register int i, j;

	scanf("%d %d", &aLength, &bLength);

	for (i = 0; i < aLength; i++) cin >> a[i];
	for (i = 0; i < bLength; i++) cin >> b[i];

	for (i = 0; i < 101; i++) {
		for (j = 0; j < 101; j++) cache[i][j] = -1;
	}
	return;
}
int max(int x, int y) {
	return x > y ? x : y;
}
int lis(int aStart, int bStart) {

	int& ret = cache[aStart + 1][bStart + 1];
	if (ret != -1) return ret;

	ret = 0;
	long long aValue = (aStart == -1 ? NEGINF : a[aStart]);
	long long bValue = (bStart == -1 ? NEGINF : b[bStart]);
	long long maxValue = max(aValue, bValue);

	for (int aNext = aStart + 1; aNext < aLength; aNext++) {
		if (maxValue < a[aNext]) {
			ret = max(ret, lis(aNext, bStart) + 1);
		}
	}
	for (int bNext = bStart + 1; bNext < bLength; bNext++) {
		if (maxValue < b[bNext]) {
			ret = max(ret, lis(aStart, bNext) + 1);
		}
	}
	return ret;
}
int  main() {

	int maxLength;
	scanf("%d", &testNum);

	for (register int testCount = 0; testCount < testNum; testCount++) {
		InputData();
		int result = lis(-1, -1);
		printf("%d\n", result);
	}

}

#endif //����Ž�� + DP : JLIS

#if 0
/*
��Ž + DP : PI
��������
1. ��ȭ�� ����°� ���� ����.
-> �������� ���� ���� �غ��� �ҵ�
�ع�
1. ���� Ž������ ó�� ���� 3,4,5 �������� �ɰ��ٰ� �����غ���.
-> ����� ���� �ʹ� ���Ƽ� Ÿ�� ����Ʈ�� �ɸ�.
2. ���� ������ ���� ���ϴ� ���̹Ƿ�, ����ȭ �����̰� ���� �κй��� ���� ����.
3. 1)���� ���� Ž���� �κй����� ������, �� �κй����� ������ �ذ��� �� ����
-> �� ������ ���, Ư�� �ε������� 3,4,5�� ������ �������� ���� ���̵� ���� ������ ���� ������ ���ؼ�
���̵� ���� ���ϴ� ����.--> min(�⺻��, ������ �� ���������� ���̵��� + ������ ���� ���������� �ּ��� ���̵���(��ͷ� �ذ�))
4. �� ������ ���� ��ȭ�� ����°� ����.
���Ӱ� �˰� �Ȱ�.
abs()�Լ��� ���밪 �������ִ� �Լ���.
*/
const int INF = 987654321;

int testNum, piLength;
string piNum;
int cache[10001];
int min(int a, int b) {
	return a > b ? b : a;
}
int classify(int begin, int length) {

	string subPiNum = piNum.substr(begin, length);

	if (subPiNum == string(subPiNum.size(), subPiNum[0])) return 1;
	//case 1;

	bool ArthProgression = true;
	for (int i = 0; i < length - 1; i++) {
		if (subPiNum[i + 1] - subPiNum[i] != subPiNum[1] - subPiNum[0]) {
			ArthProgression = false;
			break;
		}
	}
	if (ArthProgression && abs(subPiNum[1] - subPiNum[0]) == 1) return 2;//case 2;

	bool alternate = true;
	for (int i = 0; i < length; i++) {
		if (subPiNum[i] != subPiNum[i % 2]) {
			alternate = false;
			break;
		}
	}
	if (alternate) return 4; //case 3;
	if (ArthProgression) return 5; //case 4; 

	return 10;
}
int piMemorize(int begin) {

	if (begin == piNum.size()) return 0;

	int& ret = cache[begin];
	if (ret != -1) return ret;

	ret = INF;
	for (int i = 3; i <= 5; i++) {
		if (begin + i <= piNum.size()) {
			ret = min(ret, piMemorize(begin + i) + classify(begin, i));
		}
	}
	return ret;
}
void ininitialize() {
	for (register int i = 0; i < 10001; i++) cache[i] = -1;
}
int main() {

	scanf("%d", &testNum);
	for (int i = 0; i < testNum; i++) {
		cin >> piNum;
		ininitialize();

		int result = piMemorize(0);
		printf("%d\n", result);
	}
	return 0;
}
#endif //��Ž + DP : PI

#if 0
/*
��Ž + DP : Quantization
���� Ž������ ����ȭ �ĺ��� �ϳ� �� ���ؼ� ����ϸ� �ʹ� �����ð��� �ɸ�.
����� ���� ���� ���, ��ó���� ���� Ư�� ������ ����.
�� ������ ��쿡�� ������ �ϰ� ����� ���� �������� �߳����༭ �ּڰ��� ���ϸ� ��.
�ᱹ, �� ���� �������� �������� ������ ������ ���� ������ ���κ� ������ ���� ������ ������ ���� ����.
�����κ� ������ �����̵�. �׷��� DP�� ������ ����.
���Ӱ� �˰� �Ȱ�.
�κ����� ���ؼ� ���� ����� ����ð����� ��갡��.
*/
const int INF = 987654321; //��û ū ����
int length; //������ ũ��
int arr[100], partSum[100], partSquareSum[100];
int cache[100][10];
void preCalculate(void)
{
	sort(arr, arr + length); //����
	partSum[0] = arr[0];
	partSquareSum[0] = arr[0] * arr[0];

	for (int i = 1; i < length; i++)
	{
		partSum[i] = partSum[i - 1] + arr[i];
		partSquareSum[i] = partSquareSum[i - 1] + arr[i] * arr[i];
	}

}
int minDiffrence(int low, int high)
{
	//�κ����� �̿��� arr[low]...arr[high]�� �� ����
	int sum = partSum[high] - (low == 0 ? 0 : partSum[low - 1]);
	int squareSum = partSquareSum[high] - (low == 0 ? 0 : partSquareSum[low - 1]);

	//����� �ݿø��� ������ �� ������ ǥ��
	int mean = (int)(0.5 + (double)sum / (high - low + 1)); //�ݿø�
	//sum(arr[i]-mean)^2�� ������ ����� �κ������� ǥ��
	//��(arr[i]-mean)^2 = (high-low+1)*mean^2 - 2*(��arr[i])*mean + ��arr[i]^2
	int result = squareSum - (2 * mean*sum) + (mean*mean*(high - low + 1));
	return result;
}
int quantize(int from, int parts) //from��° ������ ���ڵ��� parts���� �������� ���´�
{
	//���� ���:��� ���ڸ� �� ����ȭ���� ��
	if (from == length) return 0;
	//���� ���:���ڴ� ���� ���Ҵµ� �� ���� �� ���� �� ���� ū �� ��ȯ
	if (parts == 0)	return INF;

	int &result = cache[from][parts];
	if (result != -1)return result;

	result = INF;
	//������ ���̸� ��ȭ���� ���� �ּ�ġ ã��
	for (int partSize = 1; from + partSize <= length; partSize++)
		result = min(result, minDiffrence(from, from + partSize - 1) + quantize(from + partSize, parts - 1));

	return result;
}
int main(void)
{
	int test_case;
	cin >> test_case;
	if (test_case < 0 || test_case>50)
		exit(-1);

	for (int i = 0; i < test_case; i++)
	{
		int useNum; //����� ���� ����
		cin >> length >> useNum;
		if (length < 1 || length>100 || useNum < 1 || useNum>10)
			exit(-1);

		for (int i = 0; i < length; i++)
			cin >> arr[i];

		preCalculate();
		memset(cache, -1, sizeof(cache));
		cout << quantize(0, useNum) << endl << endl;
	}

	return 0;

}
#endif //��Ž + DP : Quantization

#if 0
/*
��Ž + DP : TILING2
*/
int n;
const int MOD = 1000000007;
int cache[101];

int tiling(int n) {

	if (n <= 1) return 1;
	int& ret = cache[n];

	if (ret != -1) return ret;
	return ret = (tiling(n - 2) + tiling(n - 1)) % MOD;
}
int main(void)
{
	int test_case;
	cin >> test_case;

	for (int i = 0; i < test_case; i++)
	{
		scanf("%d", &n);
		memset(cache, -1, sizeof(int) * 101);
		int result = tiling(n);
		printf("%d\n", result);
	}
	return 0;
}
#endif //��Ž + DP : TILING2

#if 0
/*
//���� : TRIPATHCNT
�ذ�
�� ������ �ﰢ���� �ִ� ����� ������ ����� ���� �����̴�.
�׷��ٸ� �̹� ������� �ִ� ��ο��� ������ ����� ū ������ �̵��ؼ�
�ٴڱ��� ���鼭 Ž���ϸ� ��.

*/
int testCount, triangleSize, rowCount;
int triangle[100][100];
int cache[100][100];
int countCache[100][100];

int max(int left, int right) {
	return left > right ? left : right;
}
int calcMaxPath(int row, int col) {

	if (row == triangleSize - 1) return triangle[row][col];

	int& cValue = cache[row][col];
	if (cValue != -1) return cValue;

	return cValue = triangle[row][col] + max(calcMaxPath(row + 1, col), calcMaxPath(row + 1, col + 1));
}
int count(int row, int col) {

	if (row == triangleSize - 1) return 1;
	int& ret = countCache[row][col];

	ret = 0;
	if (calcMaxPath(row + 1, col) >= calcMaxPath(row + 1, col + 1)) ret += count(row + 1, col);
	if (calcMaxPath(row + 1, col) <= calcMaxPath(row + 1, col + 1)) ret += count(row + 1, col + 1);
	return ret;
}
void inititalize() {
	for (register int i = 0; i < triangleSize; i++) {
		for (register int j = 0; j < triangleSize; j++) {
			cache[i][j] = -1;
			triangle[i][j] = 0;
			countCache[i][j] = -1;
		}
	}
}
int main() {
	int result;
	int pathCount;

	scanf("%d", &testCount);
	for (register int i = 0; i < testCount; i++) {
		scanf("%d", &triangleSize);
		inititalize();
		rowCount = 1;
		for (register int i = 0; i < triangleSize; i++) {
			for (register int j = 0; j < rowCount; j++) {
				scanf("%d", &triangle[i][j]);
			}
			rowCount++;
		}
		//result = calcMaxPath(0, 0);
		pathCount = count(0, 0);
		printf("%d\n", pathCount);
	}
}
#endif //���� : TRIPATHCNT

#if 0   
/*
���� : SNAIL
Ǯ��
�� ������ �־��� ������, �����̰� �ö󰥼� �ִ��� ������ �ľ��ϴ°�
������ ����� Ȯ���� �׷��� ���� Ȯ���� �ٸ��ٴ°� ����.
����Ǯ�⿡ �ռ� �ϴ� �κй����� ������ ���� �׳� ���Ͽ��� �� ������ �ʿ�������
�ΰ��� ��찡 �ְ� �� ������ ���̶�� �칰�� �ö󰬴��� ���ö� ������ Ȯ���ؾ��Ѵ�
���������̰� �칰�� �ö󰬴��� ���ö󰬴����� ������ʰ� �ǰ� �� �̿��� �������� ��ȭ���� �����
������ �� �������� �칰�� �ö� Ȯ�� = ���� ��ͼ� 2M �ö� Ȯ�� + ��ȿͼ� 1M �ö� Ȯ���� ����.
�̸� ���� ��ȭ�� �����
SNAIL(������, ���� �ö� ����) = (0.75*snail(������ + 1, climb + 2)) + (0.25*snail(������ + 1, climb + 1));
����, ���ϰ� ��, ������ �������� �� ������ �����̹Ƿ� ��� Ȯ���� ���ϸ�ȴ�.
*/
int depth, day;
double cache[1001][2001];
double snail(int leftDay, int climb) {

	if (leftDay == day) return climb >= depth ? 1 : 0;

	double& ret = cache[leftDay][climb];
	if (ret != -1.0) return ret;

	return ret = (0.75*snail(leftDay + 1, climb + 2)) + (0.25*snail(leftDay + 1, climb + 1));
}
void init() {
	for (int i = 0; i < 1001; i++) {
		for (int j = 0; j < 2001; j++) cache[i][j] = -1.0;
	}
}
int main(void)
{
	int test_case;
	cin >> test_case;

	init();
	for (int i = 0; i < test_case; i++)
	{
		scanf("%d %d", &depth, &day);
		double result = snail(0, 0);
		printf("%0.10f\n", result);
	}
	return 0;
}
#endif  //���� : SNAIL

#if 0
/*
���� : ASYMTILING
����
������ Ÿ�ϸ� ����ó�� �� Ÿ���� �����߿��� ���Ī���� ä���ִ� Ÿ���� ���°�.
���⼭ �����Ұ� 1. ���Ī�� �� �������� 2. �� Ÿ�Ͽ��� ��Ī�ΰ͸� �������� ������
1 ���Ī�� �� ���⿡�� ��Ģ�� ���ϼ��� ���⿡ �� Ÿ�Ͽ��� ��Ī�� Ÿ���� ������ ������
�� Ÿ���� ���� ���� ���� ĳ�����ؼ� ����.
�̸� ���� �� Ÿ���� ä���ִ� �ε��������� ��� ���� �Ի�Ǿ��ִ�.
���⼭ ��Ī�� Ÿ���� ������ ��� ����?
��Ī�� ���� Ÿ���� ���̰� Ȧ��, ¦�� �϶� �ٸ���
Ȧ���϶� ��� 1���� ���� Ÿ���� ������ ���� ��Ī�̱⿡ (Ȧ�� ����-1)/2 ũ���� Ÿ�ϰ����� ����
¦���϶��� ���� �ݹ��϶�, ��� 2ũ���� Ÿ���� �� �� �̹Ƿ� �� ��츦 ���ؼ� ���ش�.
����Լ��� �� ������ ���ϴ� �Լ��� ���븸�ϸ� �ǹǷ� ����.

���⵵ : �� Ÿ���� ���ϴ� �Լ��� ���⵵�� n�̴�.
���⼭ �̰� ����� �ϹǷ� ���⵵�� n��
*/
int n;
const int MOD = 1000000007;
int cache[101];

void init() {
	for (int i = 0; i < 101; i++)  cache[i] = -1;

}
int tiling(int n) {

	if (n <= 1) return 1;
	int& ret = cache[n];

	if (ret != -1) return ret;
	return ret = (tiling(n - 2) + tiling(n - 1)) % MOD;
}
int countAsymTile(int width) {

	int ret;

	if (width % 2 == 1) {
		ret = (tiling(width) - tiling(width - 1 / 2) + MOD) % MOD;
		return ret;
	}
	else {
		ret = tiling(width);
		ret = (ret - tiling(width / 2) + MOD) % MOD;
		ret = (ret - tiling(width / 2 - 1) + MOD) % MOD;
		return ret;
	}
}
int main(void)
{
	int test_case;
	cin >> test_case;

	for (int i = 0; i < test_case; i++)
	{
		scanf("%d", &n);
		init();
		int result = countAsymTile(n);
		printf("%d\n", result);
	}
	return 0;
}

#endif //���� : ASYMTILING

#if 0
/*
���� : NUMB3RS
����
�� ������ ������ ����Ž���� ����� ������ �����̴�.
������������ ������ ��� ���� ã�� �� ã�Ƽ� ���������� ã�°�.
������ �׷��� ����Ž������ �ϸ� ������ ������ ������ �ʿ䰡 �ְ�
���� �ð��� �ɸ��Եȴ�.
�׷��� ������ DP�� �����ϱ� ���� �� �κй����� ������.
�κй����� ������ Ư�� Day�� Ư�� �������� �������� ������ Ȯ���� ���ϴ°��̴�.
�������� ���ϴ� ���� �Ʒ��� ����.

 ��� �� += search(there, day+1) / ���� here�� ����� ������ ��.
 ���⼭ search�� here, days���� ������ q�� ������ Ȯ�����̴�.

���� ���� ������ε� �� �� ������ �׷� ���, ���� ���� ���� ����������
Ȯ���� ���Ϸ��� �� ���, �ڵ尡 ��������
�׷��� ���������� ���� �Ųٷ� Ȯ���� ���ϴ°� ����(�������� �����̹Ƿ�).

���⵵ : �κ� ���� n(��������) �̰� �� day��ŭ�̶� nd�̴�.
*/
int vilageNum, day, jail, searchNum;
int vilage[50][50];
int adjVilage[50];
double cache[50][101];

double search(int curVilage, int days) {

	if (days == 0) return curVilage == jail ? 1.0 : 0.0;

	double& ret = cache[curVilage][days];
	if (ret != -1.0) return ret;

	ret = 0.0;
	for (int i = 0; i < vilageNum; i++) {
		if (vilage[curVilage][i]) {
			ret += search(i, days - 1) / adjVilage[i];
		}
	}
	return ret;
}
int main() {

	int tc;
	cin >> tc;

	for (int i = 0; i < tc; i++) {
		cin >> vilageNum >> day >> jail;

		for (int j = 0; j < vilageNum; j++) {
			for (int k = 0; k < vilageNum; k++) {
				cin >> vilage[j][k];
			}
		}

		for (int j = 0; j < vilageNum; j++) {
			for (int k = 0; k < day + 1; k++) {
				cache[j][k] = -1;
			}
		}

		memset(adjVilage, 0, sizeof(adjVilage));
		for (int j = 0; j < vilageNum; j++) {
			for (int k = 0; k < vilageNum; k++) {
				adjVilage[j] += vilage[j][k];
			}
		}

		cin >> searchNum;
		vector<int> findVilage;
		int findVilageNum;
		for (int j = 0; j < searchNum; j++) {
			cin >> findVilageNum;
			findVilage.push_back(findVilageNum);
		}

		for (int j = 0; j < searchNum; j++) {
			cout.precision(8);
			cout << search(findVilage[j], day) << " ";
		}
		cout << endl;
	}
	return 0;
}
#endif //���� : NUMB3RS

#if 0
vector<int> slice(const vector<int> &v, int a, int b) {
	return vector<int>(v.begin() + a, v.begin() + b);
}
void printPostOrder(const vector<int> &pre, const vector<int> &in) {
	const int size = pre.size();

	if (pre.empty()) return;

	int root = pre[0];
	int L = find(in.begin(), in.end(), root) - in.begin();
	int R = size - L - 1;

	printPostOrder(slice(pre, 1, L + 1), slice(in, 0, L));
	printPostOrder(slice(pre, L + 1, size), slice(in, L + 1, size));
	cout << root << ' ';

	return;
}
int main() {
	int testCase;
	cin >> testCase;

	for (int i = 0; i < testCase; i++) {
		int N, value;
		cin >> N;

		vector<int> preOrder, inOrder;
		for (int j = 0; j < N; j++) {
			cin >> value;
			preOrder.push_back(value);
		}
		for (int j = 0; j < N; j++) {
			cin >> value;
			inOrder.push_back(value);
		}
		printPostOrder(preOrder, inOrder);
		cout << endl;
	}
	return 0;
}
#endif //���� : TreeTraversal

#if 0 //Forest
/*
���� : FORTRESS
����
�� ������ Tree�� ����� �� �ִ� ���� �� �ϳ�����.
������ ������ �� Ư�� �������� �ٸ� �������� ������ �� ����ġ�� �ִ����� ������ �̴�.
�� ��, �� ������ ���̿��� "���ϰ���"�� �����ؼ� �� ������ Ʈ���� ��Ÿ�� �� �ִ�.

�� �������� ���� ���踦 ����� �� ��, �����ľ��ϴ� �ִ����� ���� ���� ���ϴ°��� ũ�� 2�Դ�.

leaf���� leaf���� �Ǵ� ��Ʈ���� leaf�����̴�.
��Ʈ���� leaf������ ���� ���� �� �ְ� leaf���� leaf ������ �ִ� ���̰� �����̴�.

leaf���� leaf������ �� Node�� Root��� ���� ��, �ڽ��� �ڽ� ��� �߿��� ���� ū ����Ʈ���� ���̸� ����
�ΰ��� ����� ���� leaf ���� leaf������ �ִ����̴�. �̰��� ��͸� �����
�� �Ʒ��� ���� �ϳ��ϳ� ���ؼ� Root ���� ���ϸ� ������ ���� �� �ִ�.
���⵵ : node n�� * n�������� ��ȸ *n�������� �ߺ� ���� - n^3 + ~
*/
#include <stdio.h>

#define MAX_NODE_NUM 110
#define MAX_CHILD_NUM 110

struct TreeNode
{
	int parent;
	int x, y, r;
	int child_num;
	TreeNode* child[MAX_CHILD_NUM];
};
TreeNode tree[MAX_NODE_NUM];
int castleNum;

void initTree(void)
{
	int i;
	int j;
	for (i = 0; i <= castleNum; i++)
	{
		tree[i].parent = -1;
		for (j = 0; j < MAX_CHILD_NUM; j++)
		{
			tree[i].child[j] = 0;
		}
		tree[i].child_num = 0;
	}
}
int sqr(int x) {
	return x * x;
}
int sqrdist(int a, int b) {
	return sqr(tree[a].y - tree[b].y) + sqr(tree[a].x - tree[b].x);
}
bool enclose(int a, int b) {
	return tree[a].r > tree[b].r && sqrdist(a, b) < sqr(tree[a].r - tree[b].r);
}
int isChild(int root, int child) {
	if (!enclose(root, child)) return false;
	for (int i = 0; i < castleNum; i++) {
		if (i != root && i != child && enclose(root, i) && enclose(i, child))
			return false;
	}
	return true;
}
void makeOrder(int root) {
	for (int castle = 0; castle < castleNum; castle++) {
		if (isChild(root, castle)) {
			tree[root].child[tree[root].child_num++] = &tree[castle];
			makeOrder(castle);
		}
	}
}
int longest;
int calcMaxHeight(TreeNode* root) {

	vector<int> height;

	for (int childNum = 0; childNum < root->child_num; childNum++) {
		height.push_back(calcMaxHeight(root->child[childNum]));
	}
	if (root->child_num == 0) return 0;

	sort(height.begin(), height.end());

	if (height.size() >= 2)
		longest = max(longest, 2 + height[height.size() - 2] + height[height.size() - 1]);

	return height.back() + 1;
}
int solve() {
	longest = 0;
	int h = calcMaxHeight(&tree[0]);

	return max(h, longest);
}
int main() {

	int test_case;

	cin >> test_case;
	for (int i = 0; i < test_case; i++) {
		cin >> castleNum;
		initTree();
		for (int j = 0; j < castleNum; j++) {
			cin >> tree[j].x >> tree[j].y >> tree[j].r;
		}
		makeOrder(0);
		int ret = solve();
		cout << ret << endl;
	}
	return 0;
}
#endif //���� : FORTRESS

#if 0
/*
���� : RUNNINGMEDIAN
	����
	�� ������ �켱���� ť�� Ȱ�� �ؼ� �߰����� ���ϴ¹����̴�.
	�־��� ������ ���̸� �� �Ǵ� �ݺ��� + 1�� 2���� ť�� ���� ������ -> �߰� ���� ���̴� ��������ϴϱ�
	�̶� �ִ�, �ּ� ������ �ݹ� �� ������ ��Ʈ�� ���ؼ� �߰� ���� ���Ѵ�.
	���⵵ : N���� �� �켱���� ť�� LOG N �׷��� N * LOG(N) ����??
*/
const int MOD = 20090711;

struct RNG
{
	int seed, a, b;
	RNG(int _a, int _b) :a(_a), b(_b), seed(1983)
	{
	}
	int next()
	{
		int result = seed;
		seed = ((seed*(long long)a) + b) % MOD;
		return result;

	}

};
int runningMedian(int n, RNG rng) {
	priority_queue<int, vector<int>, less<int>> MaxQueue;
	priority_queue<int, vector<int>, greater<int>> MinQueue;

	int ret = 0;
	for (int i = 1; i <= n; i++) {
		if (MaxQueue.size() == MinQueue.size())
			MaxQueue.push(rng.next());
		else
			MinQueue.push(rng.next());

		if (!MinQueue.empty() && !MaxQueue.empty() && (MaxQueue.top() > MinQueue.top())) {
			int minTop = MinQueue.top();
			int maxTop = MaxQueue.top();

			MinQueue.pop(); MaxQueue.pop();
			MinQueue.push(maxTop);
			MaxQueue.push(minTop);
		}
		ret = (ret + MaxQueue.top()) % MOD;
	}
	return ret;
}
int main() {

	int testCase;

	cin >> testCase;
	for (int i = 0; i < testCase; i++) {
		int N = 0, a = 0, b = 0;
		cin >> N >> a >> b;
		RNG ranGenerator(a, b);

		cout << runningMedian(N, ranGenerator) << endl;
	}
	return 0;
}
#endif //���� : RUNNINGMEDIAN

#if 0 
/*
���� : MORDOR
	����
	�� ������ �ִ� , �ּ����� ����Ʈ���� ���� Ư�� ������ ���� ���ϴ� ��.
	�ּ� , �ִ� ����Ʈ���� ������ �˸� Ǯ �� �ִ� ������.
	���⵵�� : init�ϴµ� 2n + qeury�ϴµ� 2longn�̶� n�� �����ϹǷ� O(n)��.
*/
int maxTree[400000];
int minTree[400000];

int updateTree(int* tree, int s, int e, int idx, int value, int node) {

	if (idx > e || idx < s) return tree[node];
	if (s >= e) return tree[node] = value;

	int mid = (s + e) / 2;
	int leftValue = updateTree(tree, s, mid, idx, value, node * 2 + 1);
	int rightValue = updateTree(tree, mid + 1, e, idx, value, node * 2 + 2);

	return tree[node] = max(leftValue, rightValue);
}
int queryTree(int* tree, int s, int e, int fs, int fe, int node, int mode) {

	if (fe < s || e < fs) {
		if (mode == 1) return 0;
		else return -20001;
	}
	if (fs <= s && e <= fe) return tree[node];

	int mid = (s + e) / 2;
	int leftValue = queryTree(tree, s, mid, fs, fe, node * 2 + 1, mode);
	int rightValue = queryTree(tree, mid + 1, e, fs, fe, node * 2 + 2, mode);

	return max(leftValue, rightValue);
}

int main() {

	int C, N, Q, height, a, b;

	scanf("%d", &C);
	for (int i = 0; i < C; i++) {
		scanf("%d %d", &N, &Q);
		memset(maxTree, 0, sizeof(maxTree));
		memset(minTree, 20000, sizeof(maxTree));

		for (int j = 0; j < N; j++) {
			scanf("%d", &height);
			updateTree(maxTree, 0, N - 1, j, height, 0);
			updateTree(minTree, 0, N - 1, j, -height, 0);
		}
		for (int k = 0; k < Q; k++) {
			scanf("%d %d", &a, &b);
			int maxHeight = queryTree(maxTree, 0, N - 1, a, b, 0, 1);
			int minHeight = queryTree(minTree, 0, N - 1, a, b, 0, 0);
			int result = maxHeight + minHeight;
			printf("%d\n", result);
		}
	}

}

#endif //���� : MORDOR

#if 0
#define MAX_NODE 100000

struct RMQ
{
	int n;

	vector<int> rangeMin;
	RMQ(const vector<int>& array)
	{
		n = array.size();
		int k = 1;
		while (k < n)
			k *= 2;
		k *= 2;
		rangeMin.resize(k);
		init(array, 0, n - 1, 1);
	}

	int init(const vector<int>& array, int left, int right, int node)
	{
		if (left == right)
			return rangeMin[node] = array[left];
		int mid = (left + right) / 2;
		return rangeMin[node] = min(init(array, left, mid, node * 2), init(array, mid + 1, right, node * 2 + 1));
	}

	int query(int left, int right, int node, int nodeLeft, int nodeRight)
	{
		if (right < nodeLeft || left > nodeRight)
			return 999999;
		if (left <= nodeLeft && nodeRight <= right)
			return rangeMin[node];

		int mid = (nodeLeft + nodeRight) / 2;
		return min(query(left, right, node * 2, nodeLeft, mid), query(left, right, node * 2 + 1, mid + 1, nodeRight));
	}

	int query(int left, int right)
	{
		return query(left, right, 1, 0, n - 1);
	}
};


vector<vector<int>> graph;
int no2Serial[MAX_NODE], serial2No[MAX_NODE];
int locInTrip[MAX_NODE], depth[MAX_NODE];
int nextSerial;

vector<int> tree;
int C, N, Q, a, b, ancestor;

void preorderTraverse(int here, int depths, vector<int>& trip) {

	no2Serial[here] = nextSerial;
	serial2No[nextSerial] = here;
	nextSerial++;

	depth[here] = depths;
	locInTrip[here] = trip.size();
	trip.push_back(no2Serial[here]);
	for (auto &pick : graph[here]) {
		preorderTraverse(pick, depths + 1, trip);
		trip.push_back(no2Serial[here]);
	}

}
int calcDistance(RMQ& segTree, int a, int b) {
	int fa = locInTrip[a];
	int fb = locInTrip[b];

	if (fa > fb) swap(fa, fb);
	int LCA = serial2No[segTree.query(fa, fb)];

	return depth[a] + depth[b] - 2 * depth[LCA];
}
int main() {

	vector<int> trip;

	scanf("%d", &C);

	for (int tc = 0; tc < C; tc++) {
		scanf("%d %d", &N, &Q);

		graph.clear();
		graph.resize(N + 1);

		for (int people = 1; people < N; people++) {
			scanf("%d", &ancestor);
			graph[ancestor].push_back(people);
		}
		nextSerial = 0;
		trip.clear();
		preorderTraverse(0, 0, trip);

		RMQ segTree(trip);
		for (int calcDist = 0; calcDist < Q; calcDist++) {
			scanf("%d %d", &a, &b);
			int ressult = calcDistance(segTree, a, b);
			printf("%d\n", ressult);
		}
	}
}

#endif //���� : FAMILYTREE

#if 0
struct FenwickTree {

	vector<int> tree;

	FenwickTree(int size) :tree(size + 1) {}

	int sum(int pos) {
		pos++;
		int ret = 0;
		while (pos > 0) {
			ret += tree[pos];
			pos &= (pos - 1);
		}
		return ret;
	}
	void add(int pos, int value) {
		pos++;
		while (pos < tree.size()) {
			tree[pos] += value;
			pos += (pos & -pos);
		}
	}
};
long long calcMoveCount(vector<int>& inputArray) {
	FenwickTree tree(1000000);
	long long ret = 0;

	for (int i = 0; i < inputArray.size(); i++) {
		ret += tree.sum(999999) - tree.sum(inputArray[i]);
		tree.add(inputArray[i], 1);
	}
	return ret;
}
long long mergeSort(vector<int>& input, int left, int right) {

	if (left == right) return 0; //���� ���

	int mid = (left + right) / 2;
	long long ret = mergeSort(input, left, mid) + mergeSort(input, mid + 1, right); //������

	vector<int> temp(right - left + 1); //��ģ��
	int leftStratIndex = left;
	int rightStartIndex = mid + 1;
	int tempStartIndex = 0;

	while (leftStratIndex <= mid || rightStartIndex <= right) {
		if (leftStratIndex <= mid &&
			(rightStartIndex > right || input[leftStratIndex] <= input[rightStartIndex])) {
			temp[tempStartIndex++] = input[leftStratIndex++];
		}
		else {
			ret += mid - leftStratIndex + 1;
			temp[tempStartIndex++] = input[rightStartIndex++];
		}
	}
	for (int i = 0; i < temp.size(); i++) {
		input[left + i] = temp[i];
	}
	return ret;
}
int main() {

	int test_case;
	scanf("%d", &test_case);
	for (int i = 0; i < test_case; i++)
	{
		int N;
		cin >> N;

		vector<int> A(N, 0);
		for (int i = 0; i < N; i++)
			scanf("%d", &A[i]);

		//cout << calcMoveCount(A) << endl;
		long long result = mergeSort(A, 0, N - 1);
		printf("%d\n", result);
	}

	return 0;
}
#endif //���� : MeasrueTime  

#if 1



#endif