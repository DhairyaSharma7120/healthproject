export default function handler(req, res) {
    // const body = JSON.parse(req.body)
    console.log(req.body)
    
    res.status(200).json(req.body);
  }