from .models import *

def leaf_by_url(url, id_user):
    leafs = Leaf.objects.filter(id_user=id_user)
    for leaf in leafs:
        if str(leaf.leaf_image_url == url or leaf.leaf_image_url.url == url):
            return leaf

