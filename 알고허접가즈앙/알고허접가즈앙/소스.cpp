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

//완전탐색 : 게임판 덮기.
/*
부족한점
1) input 부분에서 error check 부분이 부족했음.
해법
1) 완전 탐색을 사용해서, 각 Point 별로 검증
2) 중복의 경우, 왼쪽위에서 오른쪽 아래로 순서를 강제해서 중복 제거.
3) 덮고 빼는 동작을 숫자의 덧셈 뺄셈으로 나타내서 하나의 함수로 구현.
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
	return count; //덮어야할 곳이 3의 배수가 아니면 fail.
}
//조각의 종류.
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

		if (tx < 0 || tx >= row || ty < 0 || ty >= col) ok = 0; //범위 검사.
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
	} //시작해야할 부분 찾기.

	if (y == -1) return 1; //못찾았으면 다 덮은 거라서 return.
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
#endif //완전탐색 : 게임판 덮기

#if 0
//완전탐색 : 시계 맞추기.
/*
부족한점
1) 무엇을 할지 이해한 후, 조건으로부터 힌트 못 끌어냄.
해법
1) 완전 탐색을 사용해서, 각 스위치에 대한 모든 경우 확인.
2) 순서는 상관 없기에, 0에서 9번까지 실행.
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
	}//탈출 조건.
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
#endif //완전탐색 : 시계 맞추기.

#if 0
//완전탐색 + DP : JUMP GAME.
/*
부족한점
해법
1) 완전 탐색을 사용한다.
2) 특정 Input에 대해 항상 값은 같으므로, DP를 활용한다.
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


#endif //완전탐색 + DP : JUMP GAME

#if 0
//완전탐색 + DP : WILD Pattern
/*
부족한점
1) 각 부분문제를 어떻게 나누고, 어떤 식으로 재귀로 호출할지가 부족.
2) Vecotr에 대한 초기화가 안되서 문제를 틀렸음, 초기화 신경쓰기.
해법
1) 완전 탐색을 사용한다('*'일 경우, 완전 탐색을 사용하는데, 각 부분문제를 어떻게 나눌지가 중요).
2) 특정 Input에 대해 특정 부분문제애 대한 결과값은 같으므로, DP를 활용한다.
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

	if (patternOffset == WildPattern.size()) //pattern index가 끝까지 가면,
		return ret = (candidateOffset == inputString.size()); //candiate도 끝까지 같다! -> 일치.

	if (WildPattern[patternOffset] == '*') { //1)도중에 '*'이 나온경우;
		for (int skipCount = 0; candidateOffset + skipCount <= inputString.size(); skipCount++) {
			if (match(patternOffset + 1, candidateOffset + skipCount)) {
				return ret = 1;
			}
		}
	}
	return ret = 0; //중간에 pattern이 안 맞을때.
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
		//후보들 정렬 및 출력.
		sort(outCandiate.begin(), outCandiate.end());
		for (int j = 0; j < outCandiate.size(); j++) {
			cout << outCandiate[j] << endl;
		}
		outCandiate.clear();
	}
	return 0;
}
#endif //완전탐색 + DP : WILD Pattern

#if 0
//완전탐색 + DP : TRIANGLEPATH
/*
부족한점
1) 시간 복잡도 계산이 아직 미흡.
2) DP를 적용하지 못할 거라는 상황은 인지하지만 문제를 바꿔서 DP를 적용시키려는 사고가 부족
해법
1) 기본적으로 완전 탐색을 사용한다 -> 완전탐색 사용하면 시간초과
2) 시간 안에 들어오기 위해 입력(계산)수를 줄여야함.
3) 문제의 부분 문제는 결국, 특정 지점에서 목적지 까지의 최대 값의 Path.
4) 저걸 캐쉬 해놓고 사용

새롭게 알게 된것.
1) 최적 부분 구조 : 문제의 최적해를 구하는 데 있어, 그 부분 문제들의 각 최적해들이 모여 전체의 최적해가 되는것.
안되는 경우 : 특정 조건으로 인해, 각 부분 문제의 최적해가 문제의 조건에 위배되는 경우(값이 특정 값을 넘는다든가??)
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
#endif //완전탐색 + DP : TRIANGLEPATH

#if 0
//완전탐색 + DP : LIS
/*
부족한점

해법
1) 기본적으로 완전 탐색을 사용한다 -> 완전탐색 사용하면 시간초과
2) 현재 시점을 기점으로 앞으로의 값을 구한다는 의미에서 각 순간의 최적값을 축적.
-> 그로인해 이전의 여러 가지 경우의 수로 올라갔을 때의 변하는 값을 나중에 계산할 필요가 없음.
3) 저걸 캐쉬 해놓고 사용

새롭게 알게 된것.
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
#endif //완전탐색 + DP : LIS

#if 0
//완전탐색 + DP : JLIS
/*
부족한점
1. 점화식 세우는게 아직 부족.
해법
1. 한 줄에 대해 LIS를 구하고 그 시점에서, 다른 수열을 통해 더 구할 수 있는지 확인.
새롭게 알게 된것.
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

#endif //완전탐색 + DP : JLIS

#if 0
/*
완탐 + DP : PI
부족한점
1. 점화식 세우는게 아직 부족.
-> 수식으로 쓰는 연습 해봐야 할듯
해법
1. 완전 탐색으로 처음 부터 3,4,5 조각으로 쪼갠다고 생각해봤음.
-> 경우의 수가 너무 많아서 타임 리미트에 걸림.
2. 가장 최적의 값을 구하는 것이므로, 최적화 문제이고 최적 부분문제 적용 가능.
3. 1)번의 완전 탐색을 부분문제로 나누어, 그 부분문제당 무엇을 해결할 지 생각
-> 이 문제의 경우, 특정 인덱스에서 3,4,5로 구간을 나누었을 때의 난이도 값과 앞으로 남은 구간에 대해서
난이도 값을 구하는 것임.--> min(기본값, 현재의 이 순간에서의 난이도값 + 앞으로 남은 구간에대한 최소의 난이도값(재귀로 해결))
4. 위 식으로 인해 점화식 세우는게 가능.
새롭게 알게 된것.
abs()함수는 절대값 리턴해주는 함수임.
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
#endif //완탐 + DP : PI

#if 0
/*
완탐 + DP : Quantization
완전 탐색으로 양자화 후보를 하나 씩 구해서 계산하면 너무 오랜시간이 걸림.
경우의 수가 많을 경우, 전처리를 통해 특정 답으로 강제.
이 문제이 경우에는 정렬을 하고 비슷한 숫자 구간으로 잘나눠줘서 최솟값을 구하면 됨.
결국, 각 숫자 구간들을 최적으로 나누면 최적의 답이 나오고 각부분 문제는 이전 문제에 영향을 받지 않음.
최적부분 구조가 적용이됨. 그래서 DP도 적용이 가능.
새롭게 알게 된것.
부분합을 통해서 숫자 계산을 상수시간으로 계산가능.
*/
const int INF = 987654321; //엄청 큰 숫자
int length; //수열의 크기
int arr[100], partSum[100], partSquareSum[100];
int cache[100][10];
void preCalculate(void)
{
	sort(arr, arr + length); //정렬
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
	//부분합을 이용해 arr[low]...arr[high]의 합 구함
	int sum = partSum[high] - (low == 0 ? 0 : partSum[low - 1]);
	int squareSum = partSquareSum[high] - (low == 0 ? 0 : partSquareSum[low - 1]);

	//평균을 반올림한 값으로 이 수들을 표현
	int mean = (int)(0.5 + (double)sum / (high - low + 1)); //반올림
	//sum(arr[i]-mean)^2를 전개한 결과를 부분합으로 표현
	//∑(arr[i]-mean)^2 = (high-low+1)*mean^2 - 2*(∑arr[i])*mean + ∑arr[i]^2
	int result = squareSum - (2 * mean*sum) + (mean*mean*(high - low + 1));
	return result;
}
int quantize(int from, int parts) //from번째 이후의 숫자들을 parts개의 묶음으로 묶는다
{
	//기저 사례:모든 숫자를 다 양자화했을 때
	if (from == length) return 0;
	//기저 사례:숫자는 아직 남았는데 더 묶을 수 없을 때 아주 큰 값 반환
	if (parts == 0)	return INF;

	int &result = cache[from][parts];
	if (result != -1)return result;

	result = INF;
	//조각의 길이를 변화시켜 가며 최소치 찾음
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
		int useNum; //사용할 숫자 갯수
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
#endif //완탐 + DP : Quantization

#if 0
/*
완탐 + DP : TILING2
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
#endif //완탐 + DP : TILING2

#if 0
/*
//문제 : TRIPATHCNT
해결
이 문제는 삼각형의 최대 경로의 개수가 몇개인지 세는 문제이다.
그렇다면 이미 만들어진 최대 경로에서 이전의 경로의 큰 값으로 이동해서
바닥까지 가면서 탐색하면 됨.

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
#endif //문제 : TRIPATHCNT

#if 0   
/*
문제 : SNAIL
풀이
이 문제는 주어진 날동안, 달팽이가 올라갈수 있는지 없는지 파악하는것
하지만 비오는 확률과 그렇지 않은 확률이 다르다는게 조건.
문제풀기에 앞서 일단 부분문제를 생각해 보면 그날 당일에는 비가 오는지 않오는지의
두가지 경우가 있고 또 마지막 날이라면 우물을 올라갔는지 못올라 갔는지 확인해야한다
마지막날이고 우물을 올라갔는지 못올라갔는지는 기저사례가 되고 그 이외의 조건으로 점화식을 세운다
현재의 날 기준으로 우물에 올라갈 확률 = 오늘 비와서 2M 올라간 확률 + 비안와서 1M 올라간 확률과 같다.
이를 토대로 점화식 세우면
SNAIL(남은날, 현재 올라간 높이) = (0.75*snail(남은날 + 1, climb + 2)) + (0.25*snail(남은날 + 1, climb + 1));
또한, 리턴값 즉, 마지막 날까지의 한 조합은 독립이므로 모든 확률을 더하면된다.
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
#endif  //문제 : SNAIL

#if 0
/*
문제 : ASYMTILING
생각
기존에 타일링 문제처럼 총 타일의 개수중에서 비대칭으로 채워넣는 타일을 세는것.
여기서 생각할건 1. 비대칭을 다 세버리냐 2. 총 타일에서 대칭인것만 빼느냐의 시작점
1 비대칭을 다 세기에는 규칙과 통일성이 없기에 총 타일에서 대칭인 타일의 개수를 빼보자
총 타일의 개수 세는 것은 캐싱을해서 센다.
이를 통해 각 타일의 채워넣는 인덱스에서는 모든 값이 게산되어있다.
여기서 대칭인 타일의 개수를 어떻게 뺄까?
대칭인 것은 타일의 넓이가 홀수, 짝수 일때 다르다
홀수일때 가운데 1개의 넓이 타일이 무적권 들어가고 대칭이기에 (홀수 넓이-1)/2 크기의 타일개수만 센다
짝수일때는 완전 반반일때, 가운데 2크기의 타일이 들어갈 때 이므로 각 경우를 구해서 빼준다.
재귀함수는 총 개수를 구하는 함수를 응용만하면 되므로 생략.

복잡도 : 총 타일을 구하는 함수의 복잡도는 n이다.
여기서 이걸 상수번 하므로 복잡도는 n임
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

#endif //문제 : ASYMTILING

#if 0
/*
문제 : NUMB3RS
생각
이 문제는 기존의 완전탐색의 개념과 동일한 문제이다.
시작지점에서 인접한 모든 곳을 찾고 또 찾아서 도착지점을 찾는것.
하지만 그렇게 완전탐색으로 하면 이전의 정보를 유지할 필요가 있고
많은 시간이 걸리게된다.
그렇기 때문에 DP를 적용하기 위해 각 부분문제를 나눴다.
부분문제의 기준은 특정 Day의 특정 마을에서 도착점을 도달할 확률을 구하는것이다.
도착점을 구하는 식은 아래와 같다.

 결과 값 += search(there, day+1) / 현재 here에 연결된 마을의 수.
 여기서 search는 here, days에서 도착점 q에 도달할 확률값이다.

위와 같은 방식으로도 할 수 있지만 그럴 경우, 여러 개의 도착 마을에대한
확률을 구하려고 할 경우, 코드가 복잡해짐
그래서 도착점에서 부터 거꾸로 확률을 구하는게 용이(시작점은 고정이므로).

복잡도 : 부분 문제 n(마을개수) 이고 총 day만큼이라 nd이다.
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
#endif //문제 : NUMB3RS

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
#endif //문제 : TreeTraversal

#if 0 //Forest
/*
문제 : FORTRESS
생각
이 문제는 Tree를 사용할 수 있는 문제 중 하나였다.
문제는 성벽들 간 특정 지점에서 다른 지점으로 도달할 때 지나치는 최대한의 성벽수 이다.
이 때, 각 성벽들 사이에는 "상하관계"가 성립해서 이 구조를 트리로 나타낼 수 있다.

각 성벽들을 상하 관계를 만들고 난 후, 지나쳐야하는 최대한의 성벽 수를 구하는것은 크게 2게다.

leaf에서 leaf까지 또는 루트에서 leaf까지이다.
루트에서 leaf까지는 쉽게 구할 수 있고 leaf에서 leaf 까지의 최대 길이가 관건이다.

leaf에서 leaf까지는 각 Node를 Root라고 했을 때, 자신의 자식 노드 중에서 가장 큰 서브트리의 높이를 갖는
두개의 노드의 합이 leaf 에서 leaf까지의 최대경로이다. 이것을 재귀를 사용해
맨 아래서 부터 하나하나 구해서 Root 까지 구하면 정답을 구할 수 있다.
복잡도 : node n개 * n개에대한 순회 *n개에대한 중복 검증 - n^3 + ~
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
#endif //문제 : FORTRESS

#if 0
/*
문제 : RUNNINGMEDIAN
	생각
	이 문제는 우선순위 큐를 활용 해서 중간값을 구하는문제이다.
	주어진 수열의 길이를 반 또는 반보다 + 1한 2개의 큐로 수로 나눈다 -> 중간 값의 길이는 맞춰줘야하니까
	이때 최대, 최소 힙으로 반반 을 나누고 루트를 비교해서 중간 값을 구한다.
	복잡도 : N개의 수 우선순위 큐는 LOG N 그래서 N * LOG(N) 정도??
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
#endif //문제 : RUNNINGMEDIAN

#if 0 
/*
문제 : MORDOR
	생각
	이 문제는 최대 , 최소한의 구간트리를 만들어서 특정 구간의 합을 구하는 것.
	최소 , 최대 구간트리만 만들줄 알면 풀 수 있는 문제임.
	복잡도는 : init하는데 2n + qeury하는데 2longn이라서 n에 수렴하므로 O(n)임.
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

#endif //문제 : MORDOR

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

#endif //문제 : FAMILYTREE

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

	if (left == right) return 0; //기저 사례

	int mid = (left + right) / 2;
	long long ret = mergeSort(input, left, mid) + mergeSort(input, mid + 1, right); //나누고

	vector<int> temp(right - left + 1); //합친다
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
#endif //문제 : MeasrueTime  

#if 1



#endif