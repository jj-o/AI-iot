def solution(numbers, hand):
    answer = ''
    
    dic={1:[0,0], 2:[0,1], 3:[0,2],
         4:[1,0], 5:[1,1], 6:[1,2],
         7:[2,0], 8:[2,1], 9:[2,2],
         '*':[3,0], 0:[3,1], '#':[3,2]}
    
    left_s=dic['*']
    right_s=dic['#']
    
    for i in numbers:
        if i in [1,4,7]:
            answer+='L'
        elif i in [3,6,9]:
            answer+='R'
        if i in [2,5,8,0]:
            dis_left=

    return answer