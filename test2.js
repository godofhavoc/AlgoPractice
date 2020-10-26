var lengthOfLongestSubstring = function (s) {
    let result = 0;

    for (let i = 0; i < s.length; i++) {
        for (let j = i + 1; j < s.length; j++) {
            let temp = new Set(s.slice(i, j).split(''));
            if (temp.size === s.slice(i, j) && temp.size > result) {
                result = temp.size
            }
        }
    }

    return result
};

const sqrt = x => {
    const sqrtIter = (guess) => {
        if (goodEnough(guess)) {
            return guess;
        }
        return sqrtIter(improve(guess));
    }

    const goodEnough = (guess) => {
        if (Math.abs((guess * guess) - x) < 0.001) {
            return true;
        }
        return false;
    }

    const improve = (guess) => (((x / guess) + guess) / 2);

    return sqrtIter(x / 2, x);
}

const factorial = x => {
    let prod = 1,
        count = 1;

    while (count <= x) {
        prod = prod * count++;
    }

    return prod
}

const countChange = amt => {
    memo = Array(...Array(amt + 1)).map(() => Array(5 + 1));

    const denom = {
        1: 1,
        2: 5,
        3: 10,
        4: 25,
        5: 50
    };

    const countUtil = (amt, coins) => {
        if (amt == 0) {
            return 1;
        } else if ((amt < 0) || (coins === 0)) {
            return 0;
        } else if (memo[amt][coins]) {
            return memo[amt][coins];
        } else {
            memo[amt][coins] = countUtil(amt, coins - 1) + countUtil(amt - denom[coins], coins);
            return memo[amt][coins];
        }
    }

    return countUtil(amt, 5);
}

console.log(countChange(100));