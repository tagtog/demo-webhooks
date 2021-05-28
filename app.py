from flask import Flask, request, Response
import json
import spacy
import requests
import os
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------

# Set your tagtog credentials & project info
MY_USERNAME = os.environ['MY_TAGTOG_USERNAME']
MY_PASSWORD = os.environ['MY_TAGTOG_PASSWORD']
MY_PROJECT = os.environ['MY_TAGTOG_PROJECT']
# the project owner could be a different user, but for simplicity we assume it's the same as your username
MY_PROJECT_OWNER = MY_USERNAME

TAGTOG_DOMAIN_CLOUD = "https://tagtog.net"
TAGTOG_DOMAIN = os.environ.get('TAGTOG_DOMAIN', TAGTOG_DOMAIN_CLOUD)
# When this is false, the SSL certification will not be verified (this is useful, for instance, for self-signed localhost tagtog instances)
VERIFY_SSL_CERT = TAGTOG_DOMAIN == TAGTOG_DOMAIN_CLOUD

# -----------------------------------------------------------------------------

# API authentication
auth = requests.auth.HTTPBasicAuth(username=MY_USERNAME, password=MY_PASSWORD)

tagtog_docs_API_endpoint = f"{TAGTOG_DOMAIN}/-api/documents/v1"
tagtog_sets_API_endpoint = f"{TAGTOG_DOMAIN}/-api/settings/v1"

default_API_params = {'owner': MY_PROJECT_OWNER, 'project': MY_PROJECT}

# Parameters for the GET API call to get a document
# (see https://docs.tagtog.net/API_documents_v1.html#examples-get-the-original-document-by-document-id)
get_params_doc = {**default_API_params, **{'output': 'plain.html'}}
# Parameters for the POST API call to import a pre-annotated document
# (see https://docs.tagtog.net/API_documents_v1.html#import-annotated-documents-post)
post_params_doc = {**default_API_params, **{'output': 'null', 'format': 'anndoc'}}

# -----------------------------------------------------------------------------

# See: https://docs.tagtog.net/API_settings_v1.html#annotations-legend
def get_tagtog_anntasks_json_map():
  res = requests.get(f"{tagtog_sets_API_endpoint}/annotationsLegend", params=default_API_params, auth=auth, verify=VERIFY_SSL_CERT)
  assert res.status_code == 200, f"Couldn't connect to the given tagtog project with the given credentials (http status code {res.status_code}; body: {res.text})"
  return res.json()

# In the example of https://github.com/tagtog/demo-webhooks, we could hardcode this like:
# map_ids_to_names = {'e_1': 'PERSON', 'e_2': 'ORG', 'e_3': 'MONEY'}
# However, we use tagtog's useful API to generalize the mapping:
map_ids_to_names = get_tagtog_anntasks_json_map()
# we just invert the dictionary
map_names_to_ids = {name: class_id for class_id, name in map_ids_to_names.items()}

def get_class_id(label):
  """Translates the spaCy label id into the tagtog entity type id"""
  return map_names_to_ids.get(label, None)

# -----------------------------------------------------------------------------

# Load the spaCy trained pipeline (https://spacy.io/models/en#en_core_web_sm)
pipeline = 'en_core_web_sm'
nlp = spacy.load(pipeline)

# -----------------------------------------------------------------------------

app = Flask(__name__)
# Handle any POST request coming to the app root path

# -----------------------------------------------------------------------------

def get_entities(spans, pipeline, partId):
  """
  Translates a tuple of named entity Span objects (https://spacy.io/api/span) to list of tagtog entities (https://docs.tagtog.net/anndoc.html#ann-json)
  spans: the named entities in the spaCy doc
  pipeline: trained pipeline name
  """
  default_prob = 1
  default_part_id = partId
  default_state = 'pre-added'
  tagtog_entities = []

  for span in spans:
    class_id = get_class_id(span.label_)
    if class_id is not None:
      tagtog_entities.append({
          # entity type id
          'classId': class_id,
          'part': default_part_id,
          # entity offset
          'offsets': [{'start': span.start_char, 'text': span.text}],
          # entity confidence object (annotation status, who created it and probabilty)
          'confidence': {'state': default_state, 'who': ['ml:' + pipeline], 'prob': default_prob},
          # no entity labels (fields)
          'fields': {},
          # this is related to the kb_id (knowledge base ID) field from the Span spaCy object
          'normalizations': {}})

  return tagtog_entities


def _has_part_id(elem):
  return elem.has_attr("id")


def gen_parts_generator_over_plain_html_file(plain_html_filename):
  with open(plain_html_filename, "r") as f:
    plain_html_raw = f.read()
    return gen_parts_generator_over_plain_html(plain_html_raw)


def gen_parts_generator_over_plain_html(plain_html_raw):
  plain_html_soup = BeautifulSoup(plain_html_raw, "html.parser")

  for partElem in plain_html_soup.body.find_all(_has_part_id):
    yield partElem

# -----------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def ping():
  return "Yes, I'm here!"


@app.route('/', methods=['POST'])
def respond():
  print(request.json)
  docid = request.json.get('tagtogID')

  if docid:
    # Add the doc ID to the parameters
    get_params_doc['ids'] = docid

    get_response = requests.get(tagtog_docs_API_endpoint, params=get_params_doc, auth=auth, verify=VERIFY_SSL_CERT)
    doc_plain_html = get_response.content

    # Initialize ann.json (specification: https://docs.tagtog.net/anndoc.html#ann-json)
    annjson = {}
    # Set the document as not confirmed, an annotator will manually confirm whether the annotations are correct
    annjson['anncomplete'] = False
    annjson['metas'] = {}
    annjson['relations'] = []
    # Transform the spaCy entities into tagtog entities
    annjson['entities'] = []

    for part in gen_parts_generator_over_plain_html(doc_plain_html):
      partId = part.get('id')
      text = part.text

      # apply the spaCy model to the text
      doc = nlp(text)

      # Transform the spaCy entities into tagtog entities
      annjson['entities'] += get_entities(doc.ents, pipeline, partId)

    # Pre-annotated document composed of the content and the annotations
    files = [(docid + '.plain.html', doc_plain_html), (docid + '.ann.json', json.dumps(annjson))]

    post_response = requests.post(tagtog_docs_API_endpoint, params=post_params_doc, auth=auth, files=files, verify=VERIFY_SSL_CERT)
    print(post_response.text)

  return Response()

# -----------------------------------------------------------------------------

if __name__ == "__main__":
  app.run(host='0.0.0.0')
