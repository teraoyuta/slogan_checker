from rest_framework.response import Response
from rest_framework.decorators import api_view
from api.services.slogan_service import SloganService

@api_view(['GET'])
def check_slogan(request):
        slogan_kana = request.GET.get('slogan_kana', None)
        slogan_service = SloganService()
        sentence_distances = slogan_service.get_sentence_distance(slogan_kana)
        return Response(sentence_distances)
