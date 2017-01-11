from __future__ import print_function
import httplib2
import os
import sys

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

try:
    import argparse
    tools.argparser.add_argument('--prim', type=str)
    tools.argparser.add_argument('--arch', type=str)
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'HMLP update results in spreadsheets'

def get_credentials():
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)

    credential_path = os.path.join(credential_dir, 'sheets.googleapis.com-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME 
        if flags: 
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6 
            credentials = tools.run(flow, store)
            print('Storing credentials to ' + credential_path) 
    return credentials

#end def get_credentials()


def update_spreadsheet( service, sheetid, arch, data ):

    value_input_option = 'USER_ENTERED'
    rangeName = arch[ 0 ] + '!A2:C10'
    body = { 'values': data }

    result = service.spreadsheets().values().update(
            spreadsheetId=sheetid, range=rangeName,
            valueInputOption=value_input_option, body=body ).execute()

    result = service.spreadsheets().values().get( 
            spreadsheetId=sheetid, range=rangeName ).execute()
    values = result.get( 'values', [] )

    if not values:
        print('No data found.')
    else: 
        print('k, HMLP, Ref')
                                    
    for row in values:
        print( '%s, %s, %s' % (row[0], row[1], row[2]) )

# end def update_spreadsheet()



def update_status( service, sheetid, arch, data ):

    value_input_option = 'USER_ENTERED'
    rangeName = arch[ 0 ] + '!D2:D10'
    body = { 'values': data }

    result = service.spreadsheets().values().update(
            spreadsheetId=sheetid, range=rangeName,
            valueInputOption=value_input_option, body=body ).execute()

    result = service.spreadsheets().values().get( 
            spreadsheetId=sheetid, range=rangeName ).execute()
    values = result.get( 'values', [] )

    if not values:
        print('No data found.')
    else: 
        print('status')
                                    
    for row in values:
        print( '%s' % ( row[0] ) )

#end def update_status


def update_date( service, sheetid, arch, data ):

    value_input_option = 'USER_ENTERED'
    rangeName = arch[ 0 ] + '!D14'
    body = { 'values': data }

    result = service.spreadsheets().values().update(
            spreadsheetId=sheetid, range=rangeName,
            valueInputOption=value_input_option, body=body ).execute()

    result = service.spreadsheets().values().get( 
            spreadsheetId=sheetid, range=rangeName ).execute()
    values = result.get( 'values', [] )

    if not values:
        print('No data found.')
    else: 
        print('date')
                                    
    for row in values:
        print( '%s' % ( row[0] ) )

#end def update_status



def update_setup( service, sheetid, arch, data ):

    value_input_option = 'USER_ENTERED'
    rangeName = arch[ 0 ] + '!D16:D30'
    body = { 'values': data }

    result = service.spreadsheets().values().update(
            spreadsheetId=sheetid, range=rangeName,
            valueInputOption=value_input_option, body=body ).execute()

    result = service.spreadsheets().values().get( 
            spreadsheetId=sheetid, range=rangeName ).execute()
    values = result.get( 'values', [] )

    if not values:
        print('No data found.')
    else: 
        print('setup')
                                    
    for row in values:
        print( '%s' % ( row[0] ) )

#end def update_status



def main():

    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?'
                   'version=v4')
    service = discovery.build('sheets', 'v4', http=http, discoveryServiceUrl=discoveryUrl)

    hmlp_dir = os.getenv( 'HMLP_DIR' )
    hmlp_dat = hmlp_dir + '/build/bin/data'

    # Read all raws
    raw = [ line.strip().split( "," ) for line in open( hmlp_dat, 'r' ) ]

    print( raw )

    setup_data = []
    date_data = []
    data = []
    status_data = []
    arch = [ '', '' ]
    prim = ''

    #
    for i in range( len( raw ) ):

        if any( "@PRIM" in s for s in raw[ i ] ):
            prim = raw[ i + 1 ]

        if any( "@SETUP" in s for s in raw[ i ] ):
            setup_data.append( raw[ i + 1 ] )

        if any( "@DATE" in s for s in raw[ i ] ):
            date_data.append( raw[ i + 1 ] )

        if any( "@DATA" in s for s in raw[ i ] ):
            parse_raw = [ s.replace( " ", "" ) for s in raw[ i + 1 ] ]
            data.append( parse_raw )
        
        if any( "@STATUS" in s for s in raw[ i ] ):
            parse_raw = [ '-' + s + '0' for s in raw[ i + 1 ] ]
            status_data.append( parse_raw )

        if any( "sandybridge" in s for s in raw[ i ] ):
             arch[ 0 ] = 'x86_64/sandybridge'

    # // end for

    

    print( setup_data )
    print( date_data )
    print( data )
    print( status_data )

    print( sys.argv )

    #
    sheetid = ''
    # 
    if any( "gsks" in s for s in prim ):
        sheetid = '1hO_zLle7mf3dr7Ph02d5kUmg_xDKOpFDE6WZovb13tA'
    #
    if any( "gsknn" in s for s in prim ):
        sheetid = '1YKo2oJ1X3OT76aFl4njwEEPHcgZ6MIaQ2kFV45isH50'
    #
    if any( "conv2d" in s for s in prim ):
        sheetid = '1fc0Vmpykfo1Zxa_O3I1W_MSdOTSaor8WLBcI7UXsraY'
    #
    if any( "strassen" in s for s in prim ):
        sheetid = '1HEVv6MZyABxHMT5CzGuDsvM7pg2FjhFEVBOMFfGp_EQ'
    



    data = data[ 1:10 ]
    status_data = status_data[ 1:10 ]

    update_spreadsheet( service, sheetid, arch, data )
    update_status( service, sheetid, arch, status_data )
    update_date( service, sheetid, arch, date_data )
    update_setup( service, sheetid, arch, setup_data )

    #body = { 'values': data }

    #result = service.spreadsheets().values().update(
    #        spreadsheetId=spreadsheetId, range=rangeName,
    #        valueInputOption=value_input_option, body=body).execute()

    #result = service.spreadsheets().values().get( spreadsheetId=spreadsheetId, range=rangeName).execute()
    #values = result.get('values', [])

    #if not values:
    #    print('No data found.')
    #else: 
    #    print('Name, Major:')
    #                                
    #for row in values:
    #    print( '%s, %s, %s' % (row[0], row[1], row[2]) )



if __name__ == '__main__': 
    main()
