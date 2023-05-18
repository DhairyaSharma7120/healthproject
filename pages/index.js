import Head from 'next/head';
import styles from '../styles/Home.module.css';
import axios from 'axios';
export default function Home() {
  const fields = [
    { id: 'age', name: 'Age', type: 'text'},
    { id: 'sex', name: 'Sex', type: 'text'},
    { id: 'cp', name: 'chest pain', type: 'text'},
    { id: 'trestbps', name: 'resting blood pressure', type: 'text'},
    { id: 'chol', name: 'cholestoral', type: 'text'},
    { id: 'fbs', name: 'fasting blood sugar', type: 'text'},
    { id: 'restecg', name: 'resting electrocardiographic', type: 'text'},
    { id: 'thalach', name: 'maximum heart rate achieved', type: 'text'},
    { id: 'exang', name: 'exercise induced angina', type: 'text'},
    { id: 'oldpeak', name: 'oldpeak', type: 'text'},
    { id: 'slope', name: 'slope', type: 'text'},
    { id: 'ca', name: 'number of major vessels colored by flourosopy', type: 'text'},
    { id: 'thal', name: 'thal', type: 'text'},
    { id: 'target', name: 'target', type: 'text'},
  ]

  const handleSubmit = async (event) => {
    event.preventDefault()
    const fieldAndValues = {}
    fields?.forEach((field) => {
      fieldAndValues[field.id] = event.target[field.id].value
    })

    await axios.post('/api/uploadcsv', fieldAndValues)

  }

  
  return (
    <div className={styles.container}>
      <Head>
        <title>Get User Inpurt</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

        <form onSubmit={e => handleSubmit(e)} method='POST'>
          <div className="formContainer">
            {
              fields?.map((field) => {
                return (
                  <div className="formSection">
                    <label htmlFor={field.id}>{field.name}{`(${field.id})`}</label>
                    <input type={field.type} id={field.id} name={field.id} />
                  </div>
                )
              })
            }
          </div>
          <button className='submit-btn' type="submit">Submit</button>
        </form>

      <style jsx>{`
        form {
          width: 100%;
          display: flex;
          flex-direction: column;
          gap: 30px;
          justify-content: center;
          align-items: center;
        }
        .submit-btn {
          padding: 10px;
          width: 320px;
          cursor: pointer;
        }
        .formContainer {
          display: grid;
          width: 100%;
          grid-template-columns: repeat(3,300px);
          gap: 30px;
          justify-content: center;
        }

        input[type=text], select {
          width: 100%;
          padding: 12px 20px;
          margin: 8px 0;
          display: inline-block;
          border: 1px solid #ccc;
          border-radius: 4px;
          box-sizing: border-box;
        }
        
        input[type=submit] {
          width: 100%;
          background-color: #4CAF50;
          color: white;
          padding: 14px 20px;
          margin: 8px 0;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }
        
        input[type=submit]:hover {
          background-color: #45a049;
        }
      `}</style>

      <style jsx global>{`
        html,
        body {
          padding: 0;
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto,
            Oxygen, Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue,
            sans-serif;
        }
        * {
          box-sizing: border-box;
        }
      `}</style>
    </div>
  )
}
