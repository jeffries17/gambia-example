import React, { createContext, useContext, useState, useEffect } from 'react';
import { 
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut as firebaseSignOut,
  onAuthStateChanged,
  sendPasswordResetEmail,
  GoogleAuthProvider,
  signInWithPopup
} from 'firebase/auth';
import { doc, setDoc, getDoc, serverTimestamp } from 'firebase/firestore';
import { auth, db } from '../../config/firebase';

// Create the auth context
const AuthContext = createContext();

// Hook to use the auth context
export const useAuth = () => useContext(AuthContext);

// Provider component
export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Sign up function
  const signUp = async (email, password, name) => {
    try {
      setError(null);
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      
      // Create a user profile in Firestore
      await setDoc(doc(db, 'users', userCredential.user.uid), {
        name: name,
        email: email,
        createdAt: serverTimestamp(),
        plan: 'free',
        usageThisMonth: 0,
        reviewsAnalyzed: 0
      });
      
      return userCredential.user;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  // Sign in function
  const signIn = async (email, password) => {
    try {
      setError(null);
      return await signInWithEmailAndPassword(auth, email, password);
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  // Google sign in
  const signInWithGoogle = async () => {
    try {
      setError(null);
      const provider = new GoogleAuthProvider();
      const result = await signInWithPopup(auth, provider);
      
      // Check if user profile exists, if not create one
      const userRef = doc(db, 'users', result.user.uid);
      const userSnap = await getDoc(userRef);
      
      if (!userSnap.exists()) {
        await setDoc(userRef, {
          name: result.user.displayName || '',
          email: result.user.email,
          createdAt: serverTimestamp(),
          plan: 'free',
          usageThisMonth: 0,
          reviewsAnalyzed: 0,
          photoURL: result.user.photoURL || ''
        });
      }
      
      return result.user;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  // Reset password
  const resetPassword = async (email) => {
    try {
      setError(null);
      return await sendPasswordResetEmail(auth, email);
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  // Sign out
  const signOut = async () => {
    try {
      await firebaseSignOut(auth);
      setUserProfile(null);
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  // Fetch user profile from Firestore
  const fetchUserProfile = async (uid) => {
    try {
      const userDoc = await getDoc(doc(db, 'users', uid));
      if (userDoc.exists()) {
        setUserProfile(userDoc.data());
      } else {
        setUserProfile(null);
      }
    } catch (err) {
      console.error('Error fetching user profile:', err);
      setUserProfile(null);
    }
  };

  // Set up auth state change listener
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setCurrentUser(user);
      if (user) {
        await fetchUserProfile(user.uid);
      }
      setLoading(false);
    });

    return unsubscribe;
  }, []);

  const value = {
    currentUser,
    userProfile,
    signUp,
    signIn,
    signInWithGoogle,
    signOut,
    resetPassword,
    loading,
    error
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
};

export default AuthContext; 