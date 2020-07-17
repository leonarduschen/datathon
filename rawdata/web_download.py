import requests

locations = ['guitrancourt', 'lieusaint', 'lvs-pussay', 'parc-du-gatinais', 'arville', 'boissy-la-riviere', 'angerville-1', 'angerville-2']

for location in locations:
    url = 'https://ai4impact.org/P003/' + location + '.csv'
    r = requests.get(url, allow_redirects=True)
    open('model-a-' + location + '.csv', 'wb').write(r.content)

for location in locations:
    url = 'https://ai4impact.org/P003/' + location + '-b' + '.csv'
    r = requests.get(url, allow_redirects=True)
    open('model-b-' + location + '.csv', 'wb').write(r.content)

for location in locations:
    url = 'https://ai4impact.org/P003/historical/' + location + '.csv'
    r = requests.get(url, allow_redirects=True)
    open('historical ' + 'model-a-' + location + '.csv', 'wb').write(r.content)

for location in locations:
    url = 'https://ai4impact.org/P003/historical/' + location + '-b' + '.csv'
    r = requests.get(url, allow_redirects=True)
    open('historical ' + 'model-b-' + location + '.csv', 'wb').write(r.content)