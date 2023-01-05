def score(x):
    return 6-(max(x,1))+1

def solution(lottos, win_nums):
    i=0
    
    for x in lottos:
        if x in win_nums:
            i+=1
        return [score(i+lottos.count(0)), score(i)]
    
print(solution([44,1,0,0,31,25],[31,10,45,1,6,19]))