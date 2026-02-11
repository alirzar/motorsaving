from saveman import (connectivity, connectivity_umap, gradients)
from saveman.analyses import (reference, measures, eccentricity, adaptation, 
                             seed, behaviour, rsa, fpca)
connectivity.main()
connectivity_umap()
gradients.main()

reference.main()
measures.main()

eccentricity.main()
adaptation.main()
rsa.main
seed.main()

fpca.main()
behaviour.main()