import WebTorrent from 'webtorrent'

const client = new WebTorrent()
const magnetURI = `magnet:?xt=urn:btih:9B24D9A5EEB52BE1C493A0C81E8A046F52DE6162&dn=Call%20of%20Duty%3A%20Black%20Ops%20Cold%20War%20%26ndash%3B%20Ultimate%20Edition%20%5BDODI%20Repack%5D&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337&tr=udp%3A%2F%2Fopen.stealth.si%3A80%2Fannounce&tr=udp%3A%2F%2Ftracker.torrent.eu.org%3A451%2Fannounce&tr=udp%3A%2F%2Ftracker.bittor.pw%3A1337%2Fannounce&tr=udp%3A%2F%2Fpublic.popcorn-tracker.org%3A6969%2Fannounce&tr=udp%3A%2F%2Ftracker.dler.org%3A6969%2Fannounce&tr=udp%3A%2F%2Fexodus.desync.com%3A6969&tr=udp%3A%2F%2Fopen.demonii.com%3A1337%2Fannounce`;

client.add(magnetURI, torrent => {
  // Got torrent metadata!
  console.log('Client is downloading:', torrent.infoHash)

  for (const file of torrent.files) {
    document.body.append(file.name)
  }
})