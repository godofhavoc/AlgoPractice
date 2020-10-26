import math
import sys
from collections import defaultdict, deque
import re

def finalInstances(instances, averageUtil):
    # Write your code here
    i = 0
    while (i < len(averageUtil)):
        if averageUtil[i] < 25 and instances > 1:
            instances = math.ceil(instances / 2)
            i += 10
        elif averageUtil[i] > 60 and instances < 2e8:
            instances *= 2
            i += 10
        else:
            i += 1

    return instances

def countChange(amt):
    memo = [[-1 for x in 5] for y in amt]

    denom = {
        1: 1,
        2: 5,
        3: 10,
        4: 25,
        5: 50
    }

    ways = 0
    for coin in range(1, 6):
        if not (amt % coin):
            ways += 1

def coins(target):
    denom = [12, 10, 5, 1]

    memo = {}

    def sub_util(coin, amt):
        if (coin, amt) in memo:
            return memo[(coin, amt)]

        val = denom[coin]
        if val > amt:
            choice_take = float('inf')
        elif val == amt:
            choice_take = 1
        else:
            choice_take = 1 + sub_util(coin, amt - val)

        choice_leave = (float('inf') if coin == 0 else sub_util(coin - 1, amt))

        optimal = min(choice_take, choice_leave)
        memo[(coin, amt)] = optimal
        return optimal
    
    return sub_util(len(denom) - 1, target)

# print(coins(100))

def edit_distance(str1, str2):
    memo = [[-1 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]

    def sub_util(sub_str1, sub_str2, i, j):
        if memo[i][j] > -1:
            return memo[i][j]

        print(sub_str1, i, j)

        if i == 0:
            return j

        if j == 0:
            return i

        if sub_str1[i - 1] == sub_str2[j - 1]:
            opt = sub_util(sub_str1, sub_str2, i - 1, j - 1)
        else:
            opt = min(sub_util(sub_str1[:i] + sub_str2[j - 1] + sub_str1[i:], sub_str2, i, j - 1),
                    sub_util(sub_str1[:i] + sub_str1[i + 1:], sub_str2, i - 1, j),
                    sub_util(sub_str1[:i] + sub_str2[j - 1] + sub_str1[i + 1:], sub_str2, i - 1, j - 1)) + 1
        
        memo[i][j] = opt
        return opt

    
    return sub_util(str1, str2, len(str1), len(str2))

def edit_distance_iter(word1, word2):
    if not word1 and not word2:
        return 0
    elif not word1:
        return len(word2)
    elif not word2:
        return len(word1)

    m = len(word1)
    n = len(word2)
    d = [[float('inf') for _ in range(n + 1)] for _ in range(2)]

    for j in range(n + 1):
        d[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(n + 1):
            if j == 0:
                d[1][j] = i
            elif word1[i - 1] == word2[j - 1]:
                d[1][j] = d[0][j - 1]
            else:
                d[1][j] = min(d[1][j - 1], d[0][j - 1], d[0][j]) + 1
        d[0] = [x for x in d[1]]

    print(d)
    return d[1][n]

# print(edit_distance_iter('sunday', 'saturday'))

def solution(area):
    # Your code here
    result = []
    while (area > 0):
        n_root = math.floor(math.sqrt(area))
        result.append(int(n_root * n_root))
        read -= (n_root * n_root)

    return result

# print(solution(12))
# print(solution(15324))


def max_subarray_1(arr):
    cum_arr = [0]
    for i in range(1, len(arr)):
        cum_arr.append(arr[i] - arr[i - 1])
    
    max_sum = 0
    subarray = []
    for i in range(len(arr)):
        for j in range(1, len(arr)):
            if sum(cum_arr[i:j]) > max_sum:
                max_sum = sum(cum_arr[i:j])
                subarray = arr[i:j]

    return subarray

class MaxSubarray2:
    def max_cross_subarray(self, arr, low, mid, high):
        left_sum = right_sum = float('-inf')

        s = 0
        for i in range(mid, low - 1, -1):
            s += arr[i]
            left_sum = max(left_sum, s)

        s = 0
        for i in range(mid + 1, high + 1):
            s += arr[i]
            right_sum = max(right_sum, s)

        return left_sum + right_sum

    def sub_util(self, arr, low, high):
        if high < low:
            return float('-inf')
        if high == low:
            return arr[high]
        
        mid = int(math.floor((low + high) / 2))
        return max(self.sub_util(arr, low, mid), self.sub_util(arr, mid + 1, high), self.max_cross_subarray(arr, low, mid, high))

    def maxSubArray(self, nums):
        return self.sub_util(nums, 0, len(nums) - 1)


msub = MaxSubarray2()
# print(msub.maxSubArray([100, 113, 110, 85, 105, 102, 86, 63, 81, 101, 94, 106, 101, 79, 94, 90, 97]))
# print(msub.maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))


def salute(s):
    salutes = 0
    rcount = 0
    for i in range(len(s)):
        if s[i] == '>':
            rcount += 1
        elif s[i] == '<':
            salutes += rcount

    return salutes * 2


# print(salute('--->-><-><-->-'))
# print(salute('<<>><'))
# print(salute('>----<'))

def expand_center(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
        
    return right - left - 1


def longest_palindrome(s):
        if len(s) <= 1:
            return s

        start = end = 0
        for i in range(len(s)):
            l = max(expand_center(s, i, i),
                    expand_center(s, i, i + 1))
            if l > end - start:
                print(l)
                start = i - (l - 1) // 2
                end = i + l // 2

        return s[start:end + 1]

# print(longest_palindrome("aacabdkacaa"))
# print(longest_palindrome("babad"))

def reg_match(s, p):
    memo = {}

    def dp(i, j):
        if (i, j) not in memo:
            if j == len(p):
                ans = i == len(s)
            else:
                first_match = i < len(s) and p[j] in {s[i], '.'}
                if j + 1 < len(p) and p[j + 1] == '*':
                    ans = dp(i, j + 2) or first_match and dp(i + 1, j)
                else:
                    ans = first_match and dp(i + 1, j + 1)

            memo[i, j] = ans
        return memo[i, j]

    return dp(0, 0)

# print(reg_match('aab', 'c*a**b'))

def longest_parantheses(s):
    if len(s) < 2:
        return 0

    rcount = lcount = max_count = 0
    for i in range(len(s)):
        if s[i] == '(':
            lcount += 1
        else:
            rcount += 1

        print(lcount, rcount)

        if lcount == rcount:
            max_count = max(max_count, 2 * rcount)
        elif rcount > lcount:
            lcount = 0
            rcount = 0
    
    print(max_count)

    lcount = rcount = 0
    for i in reversed(range(len(s))):
        if s[i] == '(':
            lcount += 1
        else:
            rcount += 1

        print(lcount, rcount)

        if lcount == rcount:
            max_count = max(max_count, 2 * lcount)
        elif lcount > rcount:
            lcount = 0
            rcount = 0

    return max_count


# print(longest_parantheses("()(())"))
# print(longest_parantheses(")("))
def inside(x, y):
    if x >= 0 and x < 8 and y >= 0 and y < 8:
        return True
    return False

def minsteps(src, dest):
    # arr = [[8 * y - x for x in range(8, 0, -1)] for y in range(1, 9)]
    d = {}

    for x in range(64):
        d[x] = (x // 8, x % 8)
    
    src_pos = d[src]
    dest_pos = d[dest]

    x = [2, 2, -2, -2, 1, 1, -1, -1]
    y = [1, -1, 1, -1, 2, -2, 2, -2]

    queue = [{
        'x': src_pos[0],
        'y': src_pos[1],
        'd': 0
    }]

    visited = [[False for _ in range(8)] for _ in range(8)]

    visited[src_pos[0]][src_pos[1]] = True

    while (len(queue) > 0):
        temp = queue.pop(0)

        if (temp['x'] == dest_pos[0] and temp['y'] == dest_pos[1]):
            return temp['d']

        for i in range(8):
            l = temp['x'] + x[i]
            r = temp['y'] + y[i]

            if inside(l, r) and not visited[l][r]:
                visited[l][r] = True
                queue.append({
                    'x': l,
                    'y': r,
                    'd': temp['d'] + 1
                })

# print(minsteps(19, 42))

def wildcard_match(s, p):
    l = len(s)
    m = len(p)

    dp = [[False for _ in range(m + 1)] for _ in range(l + 1)]

    dp[0][0] = True
    for j in range(1, m + 1):
        if not p[j - 1] == '*':
            break
        dp[0][j] = True

    for i in range(1, len(s) + 1):
        for j in range(1, len(p) + 1):
            if p[j - 1] in {s[j - 1], '?'}:
                dp[i][j] = dp[i - 1][j - 1]
            elif p[j - 1] == '*':
                dp[i][j] = dp[i - 1][j] or dp[i][j - 1]

    return dp[-1][-1]

# print(wildcard_match('adceb', '?*'))

def unique_paths(m, n):
    if not m or not n:
        return 0

    dp = [1] * n

    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    
    return dp[j]

# print(unique_paths(3, 7))

def unique_paths2(obstacleGrid):
    m = len(obstacleGrid)

    if m == 0:
        return m

    n = len(obstacleGrid[0])

    if n == 0:
        return n

    if obstacleGrid[0][0] or obstacleGrid[-1][-1]:
        return 0

    dp = [[1 for _ in range(n)] for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            if obstacleGrid[i][j]:
                dp[i][j] = 0
            else:
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    
    return dp[-1][-1]

# print(unique_paths2([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
# print(unique_paths2([[], []]))

def minPathSum(grid):
    m = len(grid)
    n = len(grid[0])

    if m == 0 or n == 0:
        return 0

    for i in range(1, m):
        grid[i][0] += grid[i - 1][0]

    for j in range(1, n):
        grid[0][j] += grid[0][j - 1]

    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])

    return grid[-1][-1]

# print(minPathSum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]))

def climbStairs(n):
    a, b = 0, 1

    for _ in range(n):
        a, b = b, a + b

    return b

# print(climbStairs(5))

# monotone stack implementation
def next_greater(nums):
    ans = [-1] * len(nums)
    s = []

    for i in reversed(range(len(nums))):
        while (not len(s) == 0 and s[-1] <= nums[i]):
            s.pop()
        
        ans[i] = -1 if len(s) == 0 else s[-1]
        s.append(nums[i])

    return ans

# print(next_greater([2, 1, 2, 4, 3]))

def wait_warm(temps):
    ans = [-1] * len(temps)
    s = []

    for i in reversed(range(len(temps))):
        while (not len(s) == 0 and temps[s[-1]] <= temps[i]):
            s.pop()

        ans[i] = 0 if len(s) == 0 else (s[-1] - i)
        s.append(i)

    return ans


# print(wait_warm([73, 74, 75, 71, 69, 72, 76, 73]))

def largest_rect(heights):
    heights.append(0)
    stack = [-1]
    ans = 0

    for i in range(len(heights)):
        while heights[i] < heights[stack[-1]]:
            print(stack)
            height = heights[stack.pop()]
            width = i - stack[-1] - 1
            print(height, width)
            ans = max(height * width, ans)
        stack.append(i)

    return ans

# print(largest_rect([2, 1, 5, 6, 2, 3]))

def max_rect(matrix):
    m = len(matrix)
    n = len(matrix[0])

    height = [0] * (n + 1)
    ans = 0

    for i in range(m):
        for j in range(n):
            height[j] = height[j] + 1 if matrix[i][j] == '1' else 0
        stack = [-1]
        for j in range(n + 1):
            while height[j] < height[stack[-1]]:
                h = height[stack.pop()]
                w = j - stack[-1] - 1
                ans = max(h * w, ans)
            stack.append(j)
    
    return ans

# print(max_rect([['1', '0', '1', '0', '0'], ['1', '0', '1', '1', '1'], ['1', '1', '1', '1', '1'], ['1', '0', '0', '1', '0']]))

def is_scramble(s1, s2):
    def split(l1, r1, l2, r2):
        if r1 - l1 == 1:
            return s1[l1] == s2[l2]
        
        if sorted(s1[l1:r1]) != sorted(s2[l2:r2]):
            return False

        for i in range(1, r1 - l1):
            if split(l1, l1 + i, l2, l2 + i) and split(l1 + i, r1, l2 + i, r2) or \
                split(l1, l1 + i, r2 - i, r2) and split(l1 + i, r1, l2, r2 - i):
                return True

    return split(0, len(s1), 0, len(s2))

def numDecodings(s):
    if len(s) == 0:
        return 0

    dp = [0] * (len(s) + 1)
    dp[0] = 1

    dp[1] = 0 if s[0] == '0' else 1
    print(dp)

    for i in range(2, len(s) + 1):
        first = int(s[i - 1:i])
        second = int(s[i - 2:i])
        print(first, second, i, dp)

        if first > 0:
            dp[i] += dp[i - 1]
        if second > 9 and second < 27:
            dp[i] += dp[i - 2]
    
    print(dp)
    return dp[len(s)]

# print(numDecodings('2101'))

def is_interleave(s1, s2, s3):
    mem = {}

    def sub_util(first, second, combined):
        key = first + '_' + second + '_' + combined
        if key in mem:
            return mem[key]

        if len(first) == 0 and len(second) == 0 and len(combined) == 0:
            mem[key] = True
            return True
        elif len(combined) == 0 and (len(first) == 0 or len(second) == 0):
            mem[key] = False
            return False
        elif (len(first) == 0 and len(second) == 0) and len(combined) > 0:
            mem[key] = False
            return False
        else:
            print(first, second, combined)
            cond_one = cond_two = False
            if len(first) and first[0] == combined[0]:
                cond_one = sub_util(first[1:], second, combined[1:])
            if len(second) and second[0] == combined[0]:
                cond_two = sub_util(first, second[1:], combined[1:])
            mem[key] = cond_one or cond_two
            return cond_one or cond_two

    return sub_util(s1, s2, s3)

def is_interleave2(s1, s2, s3):
    m, n, o = len(s1), len(s2), len(s3)

    if m + n != o:
        return False

    dp = [[True for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1]
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1]

    print(dp)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            print(dp)
            dp[i][j] = (dp[i - 1][j] and s1[i - 1] == s3[i - 1 + j]) or \
                (dp[i][j - 1] and s2[j - 1] == s3[i - 1 + j])

    print(dp)
    return dp[-1][-1]

# print(is_interleave2('aabcc', 'dbbca', 'aadbbcbcac'))
# print(is_interleave2('aabcc', 'dbbca', 'aadbbbaccc'))
# print(is_interleave2('a', 'b', 'a'))
# print(is_interleave2("accccaabbccabccabbcaabaabccccbbcabcabaccccabcaccbbccaaaccab", "cbaccbcaaaaacabbbbaaaccbabbcbcbbbbbbabcbbabaababaa", "cbaccbcaaccaaccaabbcacacaabbbbbaccaaacbcbabbbcbccaabbaabbbbcccbbcabbbcbcababbcaabaabcabacaccabcaccbbccaaaccab"))
# print(is_interleave2("ab", "bc", "bcab"))


def num_distinct(s, t):
    m = len(t)
    n = len(s)

    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for j in range(n + 1):
        dp[0][j] = 1

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if t[i - 1] == s[j - 1]:
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j - 1]
            else:
                dp[i][j] = dp[i][j - 1]

    return dp[-1][-1]


# print(num_distinct('babgbag', 'bag'))

def min_triangle(triangle):
    m = len(triangle)
    dp = triangle[-1]

    for i in reversed(range(m - 1)):
        for j in range(i + 1):
            dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j]

    return dp[0]

# print(min_triangle([
#     [2],
#     [3, 4],
#     [6, 5, 7],
#     [4, 1, 8, 3]
# ]))

def inorder_traversal(root):
    if root == None:
        return True

    result = []
    stack = []

    while root != None or len(stack) != 0:
        while root:
            stack.append(root)
            root = root.left

        node = stack.pop()
        result.append(node.val)
        root = node.right

    return result

def valid_bst(root):
    if root == None:
        return True

    stack = []
    prev = None

    while root != None or len(stack) != 0:
        while root:
            stack.append(root)
            root = root.left

        node = stack.pop()
        if prev and prev >= node.val:
            return False
        prev = node.val
        root = node.right

    return True

def preorder_traversal(root):
    stack = [root]
    preorder = []

    while len(stack) != 0:
        node = stack.pop()
        preorder.append(node.val)
        
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return preorder

def componentsInGraph(gb):
    n = len(gb)
    visited = set()
    stack = []
    adj_list = defaultdict(set)
    min_path = float('inf')
    max_path = 0

    for point in gb:
        adj_list[point[0]].add(point[1])
        adj_list[point[1]].add(point[0])

    temp = 0
    for i in range(n):
        if gb[i][0] not in visited:
            stack.append(gb[i][0])

            while len(stack) != 0:
                node = stack.pop()
                temp += 1
                visited.add(node)

                for adj_node in adj_list[node]:
                    if adj_node not in visited:
                        stack.append(adj_node)

            min_path = min(min_path, temp)
            max_path = max(max_path, temp)
            temp = 0

    print('{} {}'.format(min_path, max_path))

# componentsInGraph([(1, 6), (2, 6), (2, 7), (3, 8), (4, 9)])

def numIslands(grid):
    def dfs(x, y):
        if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]) or grid[x][y] != '1':
            return 
        
        grid[x][y] = 'v'

        dfs(x + 1, y)
        dfs(x - 1, y)
        dfs(x, y + 1)
        dfs(x, y - 1)

    no_islands = 0
    
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == '1':
                no_islands += 1
                dfs(i, j)

    return no_islands

# print(numIslands([
#   ["1","1","1","1","0"],
#   ["1","1","0","1","0"],
#   ["1","1","0","0","0"],
#   ["0","0","0","0","0"]
# ]))

# print(numIslands([
#   ["1","1","0","0","0"],
#   ["1","1","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","1","1"]
# ]))

class MatchMaking:
    def get_score(self, ans1, ans2):
        score = 0

        for i in ans1:
            if ans1[i] == ans2[i]:
                score += 1

    def makeMatch(self, namesWomen, answersWomen, namesMen, answersMen, queryWoman):
        n = len(namesWomen)
        women_ans = []
        men_ans = []
        for i in range(n):
            women_ans.append(namesWomen[i] + '_' + answersWomen[i])
            men_ans.append(namesMen[i] + '_' + answersMen[i])

        women_ans.sort()
        men_ans.sort()

        score_board = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                score_board[i][j] = self.get_score(women_ans[i].split('_')[1], men_ans[i].split('_')[1])

        for i in range(n):
            if women_ans[i].split('_')[0] == queryWoman:
                return men_ans[score_board[i].index(max(score_board[i]))].split('_')[0]
            else:
                score_board[i][score_board[i].index(max(score_board[i]))] = float('-inf')

def findCircleNum(M):
    visited = []
    ans = 0
    stack = []
    
    for i in range(len(M)):
        if not i in visited:
            ans += 1
            stack.append(i)

            while len(stack) != 0:
                node = stack.pop()
                visited.append(node)
                
                for j in range(len(M[i])):
                    if not j in visited and M[node][j] == 1:
                        stack.append(j)

    return ans

# print(findCircleNum([[1,1,0],
#  [1,1,0],
#  [0,0,1]]))
# print(findCircleNum([[1,1,0],
#  [1,1,1],
#  [0,1,1]]))

def newMember(existingNames, newName):
    variations = []

    for name in existingNames:
        if newName in name:
            variations.append(newName)

    counter = 1
    for name in sorted(variations):
        if name == newName:
            newName += str(counter)
            counter += 1

    return newName


# print(newMember(["grokster2", "BrownEyedBoy", "Yoop", "BlueEyedGirl",
#                  "grokster", "Elemental", "NightShade", "Grokster1"], "grokster"))

def moneyMade(amounts, centsPerDollar, finalResult):
    revenue = 0

    revenue = sum(amounts) - amounts[finalResult]

    return (revenue * 100) - (amounts[finalResult] * centsPerDollar[finalResult])

# print(moneyMade([10, 20, 30], [20, 30, 40], 1))

def rhymeScheme(poem):
    # def islegal(word):
    #     if re.search(r'[aeiou]', word):
    #         return True
    #     elif len(word) > 2 and 'y' in word[1:-1]:
    #         return True

    def get_substr(word):
        temp = word[::]
        if word[0] == 'y':
            temp = temp[1:]

        m = re.search(r'[aeiouy]', temp)
        if m:
            return temp[m.start():]
        else:
            return word

    ans = ''
    prev_substr = None
    for i in range(len(poem)):
        if len(poem[i].rstrip()) == 0:
            ans += ' '
            print(prev_substr, 'None')
            prev_substr = None
        else:
            last_word = poem[i].rstrip().split(' ')[-1].lower()
            substr = get_substr(last_word)
            print(prev_substr, substr)

            if substr == prev_substr:
                char = ans[-1]
            elif prev_substr == None:
                if len(ans.rstrip()) > 1:
                    char_code = ord(ans.rstrip()[-1])
                    if char_code == 122:
                        char_code = 65
                    else:
                        char_code += 1
                    char = chr(char_code)
                else:
                    char = 'a'
            else:
                char_code = ord(ans[-1])
                if char_code == 122:
                    char_code = 65
                else:
                    char_code += 1
                char = chr(char_code)
            
            ans += char
            prev_substr = substr

    return ans


# print(rhymeScheme(["     ",
#                    "Measure your height",
#                    "AND WEIGHT      ",
#                    "said the doctor",
#                    "",
#                    "And make sure to take your pills",
#                    "   to   cure   your    ills",
#                    "Every",
#                    "DAY"]))

def levelOrder(root):
    ans = [[]]
    queue = [(root, 0)]
    curr = 0

    while len(queue) != 0:
        node = queue.pop(0)
        ans[node[1]].append(node[0])

        if node[0].left:
            if curr < node[1]:
                curr += 1
                ans.append([])

            queue.append((node[0].left, node[1] + 1))
        
        if node[0].right:
            if curr < node[1]:
                curr += 1
                ans.append([])

            queue.append((node[0].right, node[1] + 1))

    return ans

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isSymmetric(root):
    if not root or not (root.left and root.right):
        return True

    even = deque([root])
    odd = deque()

    while len(even) != 0 or len(odd) != 0:
        temp_stack = []

        if len(odd) == 0:
            while len(even) != 0:
                node = even.popleft()

                if node:
                    temp_stack.append(node.val)
                    odd.append(node.left)
                    odd.append(node.right)

        elif len(even) == 0:
            while len(odd) != 0:
                node = odd.popleft()

                if node:
                    temp_stack.append(node.val)
                    even.append(node.left)
                    even.append(node.right)

        print(temp_stack)
        if len(temp_stack) > 1:
            mid = len(temp_stack) // 2
            if not (temp_stack[:mid] == list(reversed(temp_stack[mid:]))):
                return False

    return True

def testing():
    # case = [1, 2, 2, 3, 4, 4, 3]
    case = [1, 2, 2, None, 3, None, 3]
    
    def insert_level(arr, root, i):
        if i < len(arr):
            temp = TreeNode(arr[i])
            root = temp

            root.left = insert_level(arr, root.left, 2 * i + 1)
            root.right = insert_level(arr, root.right, 2 * i + 2)

        return root

    root = None
    root = insert_level(case, root, 0)
    print(isSymmetric(root))

# testing()

def find_max_crossing_subarray(A, low, mid, high):
    left_sum = float('-inf')
    s = 0
    max_right = high
    max_left = low

    for i in reversed(range(low, mid + 1)):
        s += A[i]
        if s > left_sum:
            left_sum = s
            max_left = i

    right_sum = float('-inf')
    s = 0

    for j in range(mid + 1, high):
        s += A[j]
        if s > right_sum:
            right_sum = s
            max_right = j
    
    return (max_left, max_right, left_sum + right_sum)

def find_max_subarray(A, low, high):
    if high == low:
        return (low, high, A[low])
    else:
        mid = (low + high) // 2

        (left_low, left_high, left_sum) = find_max_subarray(A, low, mid)
        (right_low, right_high, right_sum) = find_max_subarray(A, mid + 1, high)
        (cross_low, cross_high, cross_sum) = find_max_crossing_subarray(A, low, mid, high)

        if left_sum >= right_sum and left_sum >= cross_sum:
            return (left_low, left_high, left_sum)
        elif right_sum >= left_sum and right_sum >= cross_sum:
            return (right_low, right_high, right_sum)
        else:
            return (cross_low, cross_high, cross_sum)

def maxProfit(prices):
    price_changes = [0]

    for i in range(1, len(prices)):
        price_changes.append(prices[i] - prices[i - 1])

    low, high, _ = find_max_subarray(price_changes, 0, len(price_changes) - 1)
    if low == high == 0:
        return 0
    if low == high == len(prices) - 1:
        return prices[-1] - prices[0]

    return prices[high] - prices[low - 1]


print(maxProfit([1, 2, 4]))
