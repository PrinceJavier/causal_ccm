from distutils.core import setup
setup(
  name = 'causal_ccm',         # How you named your package folder (MyLib)
  packages = ['causal_ccm'],   # Chose the same as "name"
  version = '0.3.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'implementation of convergent cross mapping by Sugihara et al (2012)',   # Give a short description about your library
  long_description='# causal_ccm\nPackage implementing Convergent Cross Mapping for causality inference in dynamical systems as defined by [Sugihara et al (2012)](https://science.sciencemag.org/content/338/6106/496)\n\nSee `usage_sample.ipynb` for an example usage.\n\n## To install\n`pip install causal-ccm`\n\n## To use\nSay we want to check if X drives Y. We first define `ccm` using:\n* `X` and `Y` - time series data\n* `tau` - time lag (if `tau=1` we get `[t, t-1, t-2...]` as our shadow manifold embedding\n* `E` - embedding dimension (default=2) for the shadow manifold\n* `L` - time horizon to consider, defaults at length of time series X\n\nWe define `ccm`:\n<br>`ccm1 = ccm(X, Y, tau, E, L) # define ccm with X, Y time series `\n\nWe check the strength of causality measured as correlation in prediction vs true (see Sugihara (2012))\n<br>`ccm1.causality()`\n\nWe can visualize cross mapping between manifolds of X and Y\n<br>`ccm1.visualize_cross_mapping()`\n\nWe visualize correlation of X->Y\n<br>We stronger correlation = stronger causal relationship\n<br>`ccm1.plot_ccm_correls()`\n\n\n\n',
  long_description_content_type="text/markdown",
  author = 'Prince Javier',                   # Type in your name
  author_email = 'othepjavier@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/PrinceJavier/causal_ccm',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/PrinceJavier/causal_ccm/archive/v_03.tar.gz',    # I explain this later on
  keywords = ['causality', 'dynamical systems', 'complex systems'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'matplotlib',
          'seaborn',
          'scipy'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
