# 울산항에 어떤 품목이 얼마나 왔는지 있음
# 입항년도, 적하국가명, 양하국가명, 품목코드, 품명, 중량통, 용적톤 남기고 제거. 해당 항목이 비어있는 줄 제거
# (선박의 운임 거리*운임톤*위험율)/수하물 총 양
# 전체 물동량 추이와 PESIF 점수 추이를 그래프로 나타냄

import math

class GeoUtil:
    def degree2radius(degree):
        return degree*(math.pi/180)

    def get_harversion_distance(x1,y1,x2,y2,round_decimal_digits=5):
        # (x1, y1)과 (x2, y2) 점의 거리를 반환
        if x1 is None or y1 is None or y2 is None:
            return None
        
        R=6371
        dLon=GeoUtil.degree2radius(x2-x1)
        dLat=GeoUtil.degree2radius(y2-y1)

        a=math.sin(dLat/2)*math.sin(dLat/2)+

    # haversion formula 로 경도 계산
    # 유클리안 Formula로 거리 반환